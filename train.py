import os
import time
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from utils.verbalizer import Verbalizer
from utils.metirc_utils import ClassEvaluator
from utils.common_utils import *
from data_handle.data_loader import *
from ptune_config import *
from tqdm import tqdm


# todo 创建配置对象
pc = ProjectConfig()

# 模型评估
def evaluate_model(model, metric, data_loader, tokenizer, verbalizer):
    """
    在测试集上评估当前模型的训练效果。
    参数:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    # todo 设置评估模式
    model.eval()
    # 每次重置评估指标
    metric.reset()
    # 禁用梯度计算:节省内存,提高推理速度
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            # TODO 兼容不需要 token_type_id 的模型, 比如:Roberta-Base
            if 'token_type_ids' in batch:
                logits = model(input_ids=batch['input_ids'].to(pc.device),
                               attention_mask=batch['attention_mask'].to(pc.device),
                               token_type_ids=batch['token_type_ids'].to(pc.device)).logits
            else:
                logits = model(input_ids=batch['input_ids'].to(pc.device),
                               attention_mask=batch['attention_mask'].to(pc.device)).logits
            # 将张量数据转换为Python列表格式
            mask_labels = batch['mask_labels'].numpy().tolist()  # (batch, label_num)
            for i in range(len(mask_labels)):  # 去掉label中的[PAD] token
                while tokenizer.pad_token_id in mask_labels[i]:
                    mask_labels[i].remove(tokenizer.pad_token_id)
            mask_labels = [''.join(tokenizer.convert_ids_to_tokens(t)) for t in mask_labels]  # id转文字
            predictions = convert_logits_to_ids(logits,
                                                batch['mask_positions']).cpu().numpy().tolist()  # (batch, label_num)
            # todo 找到子标签对应的主标签
            predictions = verbalizer.batch_find_main_label(predictions)  # 找到子label属于的主label
            predictions = [ele['label'] for ele in predictions]
            # todo 调用add_batch转换
            metric.add_batch(pred_batch=predictions, gold_batch=mask_labels)
    # TODO 调用compute()生成完整的评估指标
    eval_metric = metric.compute()
    # TODO 最后一定记得切换回训练模式
    model.train()
    # todo 返回评估结果
    return eval_metric['accuracy'], eval_metric['precision'], \
           eval_metric['recall'], eval_metric['f1'], \
           eval_metric['class_metrics']


def model2train():
    # TODO 1.获取数据加载器
    train_dataloader, dev_dataloader = get_data()


    # TODO 2.加载预训练模型和分词器
    model = AutoModelForMaskedLM.from_pretrained(pc.pre_model)
    # print(f'预训练模型带MLM头的--》{model}')

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    # 调用verbalizer工具类中的Verbalizer对象
    verbalizer = Verbalizer(verbalizer_file=pc.verbalizer,
                            tokenizer=tokenizer,
                            max_label_len=pc.max_label_len
                            )
    # print(f'verbalizer--》{verbalizer.label_dict}')

    # TODO 3.创建损失函数和评估器
    # 定义交叉熵损失函数，用于多分类任务的损失计算
    criterion = torch.nn.CrossEntropyLoss()
    # 初始化训练过程中的损失值列表，用于记录每个step的loss
    loss_list = []
    # todo 初始化分类评估器，用于计算模型评估指标
    metric = ClassEvaluator()

    # TODO 4.创建AdamW优化器，使用分组的参数配置
    # 定义不需要权重衰减的参数名称列表
    # print([n for n, p in model.named_parameters()])
    no_decay = ["bias", "LayerNorm.weight"]

    # 将模型参数分为两组：需要权重衰减的参数和不需要权重衰减的参数
    optimizer_grouped_parameters = [
        # 第一组：不包含bias和LayerNorm.weight的参数，应用权重衰减
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        # 第二组：包含bias和LayerNorm.weight的参数，不应用权重衰减
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # AdamW通过解耦权重衰减解决了Adam因自适应学习率导致的正则化效果弱化问题
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    # 将模型移动到指定的设备上（CPU或GPU）
    model.to(pc.device)



    # TODO 5.学习率预热调度器
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    # 指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    # 计算预热阶段的训练步数，用于学习率预热策略
    warm_steps = int(pc.warmup_ratio * max_train_steps)  # 预热阶段的训练步数
    """
    创建线性学习率调度器，实现学习率预热和线性衰减
        name: 调度器类型，'linear'表示线性衰减
        optimizer: 优化器对象，用于调整学习率
        num_warmup_steps: 预热阶段步数，在此期间学习率从0线性增长到初始值
        num_training_steps: 总训练步数，用于计算学习率衰减率
    """
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    # TODO 6.训练和评估
    print('开始训练~')
    # 记录训练开始时间，用于计算训练耗时
    tic_train = time.time()
    # 初始化全局步数和最佳F1分数，用于训练过程控制和模型保存
    global_step, best_f1 = 0, 0

    # TODO 外层循环，用于迭代训练轮数
    for epoch in range(pc.epochs):
        # TODO 内层循环，用于迭代训练数据
        for batch in tqdm(train_dataloader):
            # print(f'batch--》{batch}')
            # TODO 兼容不需要 token_type_id 的模型, 比如: Roberta-Base
            if 'token_type_ids' in batch:
                result = model(input_ids=batch['input_ids'].to(pc.device),
                               token_type_ids=batch['token_type_ids'].to(pc.device),
                               attention_mask=batch['attention_mask'].to(pc.device))
            else:
                result = model(input_ids=batch['input_ids'].to(pc.device),
                               attention_mask=batch['attention_mask'].to(pc.device))
            # todo  从模型返回中获取logits
            logits = result.logits
            # print(f'logits->{logits.shape}') # [8, 256, 21128]
            # 获取真实标签并转换列表
            mask_labels = batch['mask_labels'].numpy().tolist()
            # print(f'mask_labels--》{mask_labels}') # [[6132, 3302], ... [2398, 3352], ]
            # todo 调用verbalizer工具类获取子标签
            sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
            # print(f'sub_labels-->{sub_labels}') # [{'sub_labels': ['衣服'], 'token_ids': [[6132, 3302]]},...]
            sub_labels = [ele['token_ids'] for ele in sub_labels]
            # print(f'sub_labels-->{sub_labels}')  # [[[6132, 3302]],...,[[3819, 3861]]]

            # todo 调用mlm_loss工具类获取损失值
            loss = mlm_loss(logits,
                            batch['mask_positions'].to(pc.device),
                            sub_labels,
                            criterion,
                            pc.device,
                            )
            # print(f'计算损失值--》{loss}')  # 举例: 2.4080650806427
            # todo 将损失值添加到损失列表中,并且全局步数加1
            loss_list.append(loss)
            global_step += 1
            # TODO 模型反向传播，更新模型参数
            optimizer.zero_grad()
            loss.backward()  # 自动微分,求导数(梯度)
            optimizer.step()  # 优化器更新参数: w新 = w旧-lr*梯度
            lr_scheduler.step()  # 学习率调度器更新lr

            # todo 打印训练日志
            if global_step % pc.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                #     writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, pc.logging_steps / time_diff))
                tic_train = time.time()
            # TODO  模型验证并保存模型
            if global_step % pc.valid_steps == 0:
                acc, precision, recall, f1, class_metrics = evaluate_model(model,
                                                                           metric,
                                                                           dev_dataloader,
                                                                           tokenizer,
                                                                           verbalizer
                                                                )

                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}')
                    best_f1 = f1
                    cur_save_dir = os.path.join(pc.save_dir, "model_best2")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    # TODO 保存model和tokenizer
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()
    print('训练结束')


if __name__ == '__main__':
    # TODO 模型训练
    model2train()
