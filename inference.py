"""
加载预训练好的模型并测试效果。
"""
import time
from typing import List

import torch
from rich import print
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils.verbalizer import Verbalizer
from data_handle.data_preprocess import convert_example
from utils.common_utils import convert_logits_to_ids
from ptune_config import ProjectConfig



# TODO 创建配置对象
cfg = ProjectConfig()
# 设备配置：cuda:0表示使用第一个GPU进行计算
device = cfg.device
print(f'当前device: {device}')
model_path = 'checkpoints/model_best'
# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 加载预训练的掩码语言模型
model = AutoModelForMaskedLM.from_pretrained(model_path)
# 将模型移动到指定设备并设置为评估模式
model.to(device).eval()

# 初始化Verbalizer对象，用于处理标签词汇映射
verbalizer = Verbalizer(
    verbalizer_file='data/verbalizer.txt',  # 标签词汇映射文件路径
    tokenizer=tokenizer,  # 分词器对象
    max_label_len=cfg.max_label_len  # 标签最大长度
)


def inference(contents: List[str]):
    """
    推理函数，输入原始句子，输出mask label的预测值。
    参数:
        contents (List[str]): 描原始句子列表。
    """
    # todo 禁用梯度计算: 节省内存,提升速度
    with torch.no_grad():
        start_time = time.time()
        examples = {'text': contents}
        tokenized_output = convert_example(
            examples,
            tokenizer,
            max_seq_len=128,
            max_label_len=cfg.max_label_len,
            train_mode=False,  # 当前是预测阶段,此处一定要改为的False
            return_tensor=True,
            # todo 传入软提示中伪token数量
            p_embedding_num=cfg.p_embedding_num
        )
        logits = model(input_ids=tokenized_output['input_ids'].to(device),
                       token_type_ids=tokenized_output['token_type_ids'].to(device),
                       attention_mask=tokenized_output['attention_mask'].to(device)).logits
        predictions = convert_logits_to_ids(logits, tokenized_output[
            'mask_positions']).cpu().numpy().tolist()  # (batch, label_num)
        # TODO 找到子label属于的主label,如果没有直接找到,hard_mapping=True默认最相似的主label
        predictions = verbalizer.batch_find_main_label(predictions)
        # todo 获取对应的标签
        predictions = [ele['label'] for ele in predictions]
        # 计算使用时间
        used = time.time() - start_time
        print(f'Used {used}s.')
        # todo 返回结果
        return predictions


if __name__ == '__main__':
    print("针对下面的文本评论，请分别给出对应所属类别：")
    contents = [
        '天台很好看，躺在躺椅上很悠闲，因为活动所以我觉得性价比还不错，适合一家出行，特别是去迪士尼也蛮近的，下次有机会肯定还会再来的，值得推荐',
        '环境，设施，很棒，周边配套设施齐全，前台小姐姐超级漂亮！酒店很赞，早餐不错，服务态度很好，前台美眉很漂亮。性价比超高的一家酒店。强烈推荐',
        "物流超快，隔天就到了，还没用，屯着出游的时候用的，听方便的，占地小",
        "福行市来到无早集市，因为是喜欢的面包店，所以跑来集市看看。第一眼就看到了，之前在微店买了小刘，这次买了老刘，还有一直喜欢的巧克力磅蛋糕。好奇老板为啥不做柠檬磅蛋糕了，微店一直都是买不到的状态。因为不爱碱水硬欧之类的，所以期待老板多来点其他小点，饼干一直也是大爱，那天好像也没看到",
        "服务很用心，房型也很舒服，小朋友很喜欢，下次去嘉定还会再选择。床铺柔软舒适，晚上休息很安逸，隔音效果不错赞，下次还会来",
        "这次来深圳的玩的非常开心,是一个美好的旅程",
        "这个榴莲水果,虽然闻的臭,长的难看,但是味道很棒,很容易上瘾,下次买来吃",
        "上衣太小了,穿上很紧,而且颜色也不正"
    ]
    # TODO 调用api获取结果
    res = inference(contents)
    # todo 构建结果字典
    new_dict = {}
    for i in range(len(contents)):
        new_dict[contents[i]] = res[i]
    print(new_dict)
