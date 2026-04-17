# coding:utf-8
import torch


class ProjectConfig(object):
    def __init__(self):
        # 设置训练设备，优先使用CUDA GPU，否则使用CPU
        # Windows/Linux系统使用cuda:0，Mac系统可改为mps:0
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # windows电脑/linux服务器
        # self.device = "mps:0" # MAC电脑
        # 项目根目录路径
        self.root = r'F:\00_线上AI大模型授课资料\博学谷_AI大模型7期\llm_project\02_新零售项目\02_Bert_P-Tuning微调'
        # 预训练模型配置
        self.pre_model = self.root + '/bert-base-chinese'  # 使用中文BERT基础模型

        # 数据集路径配置
        self.train_path = self.root + '/data/train.txt'  # 训练集文件路径
        self.dev_path = self.root + '/data/dev.txt'  # 验证集文件路径
        self.verbalizer = self.root + '/data/verbalizer.txt'  # Verbalizer标签映射文件路径

        # TODO 伪token嵌入数量
        self.p_embedding_num = 6
        # 模型训练超参数配置
        self.max_seq_len = 512  # 输入序列最大长度
        self.batch_size = 8  # 训练批次大小
        self.learning_rate = 5e-5  # 学习率
        self.weight_decay = 0  # 权重衰减系数
        self.warmup_ratio = 0.06  # 学习率预热比例
        self.max_label_len = 2  # 标签最大长度(也就是mask掩码长度)
        self.epochs = 10  # 训练轮数


        # 训练过程控制参数
        self.logging_steps = 5  # 日志打印间隔步数
        self.valid_steps = 20  # 验证间隔步数

        # 模型保存配置
        self.save_dir = self.root +'/checkpoints'  # 模型检查点保存目录


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.p_embedding_num)
    print(pc.pre_model)
