#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Author ：Cheery Cherry
# @Date ：2025/4/27 星期日 20:03
import pickle
import sys
import os


class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def describe(self):
        return f"{self.name} is a {self.species}."


# 创建 Animal 实例
my_animal = Animal("Leo", "Lion")
print(my_animal.describe())  # 输出: Leo is a Lion.


# 定义一个函数计算斐波那契数列
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence


# 调用斐波那契函数并打印结果
fib_result = fibonacci(10)
print(f"斐波那契数列前 10 项: {fib_result}")

# 检查当前工作目录
current_directory = os.getcwd()
print(f"当前工作目录: {current_directory}")

# 创建一个文件并写入内容
file_name = "example.txt"
with open(file_name, "w") as file:
    file.write("输入内容。")
    file.write("该包含多行内容。")
print(f"文件 '{file_name}' 已创建并写入内容。")

# 读取文件内容并打印
with open(file_name, "r") as file:
    content = file.read()
    print(f"文件内容:{content}")

# 删除文件
os.remove(file_name)
print(f"文件 '{file_name}' 已删除。")

# 定义一个学生分数
student_scores = {
    "Alice": 95,
    "Bob": 88,
    "Charlie": 72,
    "David": 91
}

# 打印字典内容
print(f"学生分数: {student_scores}")

# 添加新学生分数
student_scores["Eve"] = 89
print(f"添加新学生后的分数: {student_scores}")

# 计算平均分数
average_score = sum(student_scores.values()) / len(student_scores)
print(f"平均分数: {average_score:.2f}")

# 打印最终信息
print("脚本执行完成。")
try:
    pass
except Exception:
    pass
a = sys.version > "3.9"
print(a)
