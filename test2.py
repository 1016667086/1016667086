a = "[MASK]"
b = a * 2
print(b)
print('================================================')
# todo numpy的where()方法
import numpy as np

a = np.array([1.0, 2.0, 3.0])
print(a == 3.0)
print(np.where(a < 3.0))
print('================================================')


def fun(a, b):
    print(f'a-->{a}')
    print(f'b-->{b}')
    return a + b


c = fun(1, 2)
print(c)
print('-------------------------------')
'''
partial函数解析
partial(fun, *args, **keywords)
'''
from functools import partial

# #
fun2 = partial(fun, b=2)
print(fun2(1))
print(fun2(2))


def multiply(x, y, z):
    return x * y * z


# 创建一个新函数，固定 x=2, y=3
double_triple = partial(multiply, 2, 3)

# 调用时只需传入 z
result = double_triple(4)  # 相当于 multiply(2, 3, 4)
print(result)  # 输出 24
print('================================================')

# todo zip打包
list1 = [[[1, 2]], [[2, 3]]]  # -->[2,1, 2]
list2 = [[2], [3]]  # -->[2,1]
list3 = [[[3, 4]], [[5, 6]], [[3, 5]]]  # --->[3, 1,2]

print(list(zip(list1, list2, list3)))

print('================================================')
from datasets import Dataset

# 创建一个简单数据集
data = {"text": ["Hello world", "How are you?", "I am fine"]}
dataset = Dataset.from_dict(data)

# 定义处理函数
def add_prefix(example):
    example["text"] = "前缀" + example["text"]
    return example

# 应用函数
new_dataset = dataset.map(add_prefix)
print(new_dataset["text"])
# 输出: ['Prefix: Hello world', 'Prefix: How are you?', 'Prefix: I am fine']

print('================================================')
# todo 绝对路径和拼接路径
import os

base_path = os.path.abspath('01_Bert_PET微调/data_handle')
print(f'base_path-->{base_path}')
cur_save_dir = os.path.join(base_path, '%s' % "ai1")
print(cur_save_dir)
a = os.path.join(cur_save_dir)
print(a)
print('================================================')
# todo argmax()方法
import torch

a = torch.tensor([[1, 2, 0, 10, 9, ],
                  [2, 4, 6, 1, 0]])
b = a.argmax(dim=-1)
print(b)
print('================================================')
list1 = [1, 2]
max_len = 3
a = list1[:max_len]
if len(a) < max_len :
    b = a + [0] * (max_len - len(a))
    print(b)