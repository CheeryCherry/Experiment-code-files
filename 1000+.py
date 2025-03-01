import os
import random
import string
import json
import time
from collections import Counter
import os
import json
import pandas as pd
import pickle
import ssl
from datetime import datetime
# 学生类
class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.scores = {}  # 科目: 分数

    def add_score(self, subject, score):
        """添加成绩"""
        if subject in self.scores:
            print(f"{self.name} 的 {subject} 成绩已存在，将被更新。")
        self.scores[subject] = score
        print(f"{self.name} 的 {subject} 成绩已添加/更新。")

    def get_average_score(self):
        """计算平均分"""
        if not self.scores:
            return 0
        return sum(self.scores.values()) / len(self.scores)

    def __str__(self):
        """返回学生信息的字符串表示"""
        scores_str = ", ".join([f"{subject}: {score}" for subject, score in self.scores.items()])
        return f"学生: {self.name} (学号: {self.student_id}), 成绩: {scores_str}, 平均分: {self.get_average_score():.2f}"

# 学生管理系统类
class StudentManager:
    def __init__(self):
        self.students = {}  # 学号: Student 对象

    def add_student(self, name, student_id):
        """添加学生"""
        if student_id in self.students:
            print(f"学号 {student_id} 已存在，无法添加。")
            return
        self.students[student_id] = Student(name, student_id)
        print(f"学生 {name} (学号: {student_id}) 已添加。")

    def add_score(self, student_id, subject, score):
        """录入成绩"""
        if student_id not in self.students:
            print(f"学号 {student_id} 不存在。")
            return
        self.students[student_id].add_score(subject, score)

    def query_student(self, student_id):
        """查询学生信息"""
        if student_id not in self.students:
            print(f"学号 {student_id} 不存在。")
            return
        print(self.students[student_id])

    def show_all_students(self):
        """显示所有学生信息"""
        if not self.students:
            print("没有学生信息。")
            return
        for student in self.students.values():
            print(student)

    def save_to_file(self, filename="students.json"):
        """保存学生数据到文件"""
        data = {}
        for student_id, student in self.students.items():
            data[student_id] = {
                "name": student.name,
                "scores": student.scores
            }
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"学生数据已保存到文件 {filename}。")

    def load_from_file(self, filename="students.json"):
        """从文件加载学生数据"""
        if not os.path.exists(filename):
            print(f"文件 {filename} 不存在。")
            return
        with open(filename, "r") as file:
            data = json.load(file)
        for student_id, info in data.items():
            student = Student(info["name"], student_id)
            for subject, score in info["scores"].items():
                student.add_score(subject, score)
            self.students[student_id] = student
        print(f"学生数据已从文件 {filename} 加载。")

# 主菜单
def main_menu():
    manager = StudentManager()
    while True:
        print("\n===== 学生成绩管理系统 =====")
        print("1. 添加学生")
        print("2. 录入成绩")
        print("3. 查询学生成绩")
        print("4. 显示所有学生信息")
        print("5. 保存数据到文件")
        print("6. 从文件加载数据")
        print("7. 退出")
        choice = input("请选择操作: ")

        if choice == "1":
            name = input("请输入学生姓名: ")
            student_id = input("请输入学生学号: ")
            manager.add_student(name, student_id)

        elif choice == "2":
            student_id = input("请输入学生学号: ")
            subject = input("请输入科目: ")
            score = float(input("请输入成绩: "))
            manager.add_score(student_id, subject, score)

        elif choice == "3":
            student_id = input("请输入学生学号: ")
            manager.query_student(student_id)

        elif choice == "4":
            manager.show_all_students()

        elif choice == "5":
            filename = input("请输入保存文件名（默认: students.json）: ") or "students.json"
            manager.save_to_file(filename)

        elif choice == "6":
            filename = input("请输入加载文件名（默认: students.json）: ") or "students.json"
            manager.load_from_file(filename)

        elif choice == "7":
            print("退出系统。")
            break

        else:
            print("无效选择，请重试。")

# 定义一个空列表来存储任务
tasks = []

# 添加任务的函数
def add_task():
    task_name = input("请输入任务名称: ")
    task = {"name": task_name, "completed": False}  # 创建一个任务字典
    tasks.append(task)  # 将任务添加到任务列表
    print(f"任务 '{task_name}' 已添加！")


# 查看任务列表的函数
def view_tasks():
    if not tasks:
        print("当前没有任务。")
    else:
        print("任务列表:")
        for index, task in enumerate(tasks):
            status = "已完成" if task["completed"] else "未完成"
            print(f"{index + 1}. {task['name']} - {status}")


# 标记任务为完成的函数
def complete_task():
    view_tasks()
    try:
        task_number = int(input("请输入要标记为完成的任务编号: ")) - 1
        if 0 <= task_number < len(tasks):
            tasks[task_number]["completed"] = True
            print(f"任务 '{tasks[task_number]['name']}' 已标记为完成！")
        else:
            print("无效的任务编号。")
    except ValueError:
        print("请输入有效的数字。")


# 删除任务的函数
def delete_task():
    view_tasks()
    try:
        task_number = int(input("请输入要删除的任务编号: ")) - 1
        if 0 <= task_number < len(tasks):
            removed_task = tasks.pop(task_number)
            print(f"任务 '{removed_task['name']}' 已删除！")
        else:
            print("无效的任务编号。")
    except ValueError:
        print("请输入有效的数字。")


# ssl.wrap_socket(ssl_version=ssl.PROTOCOL_TLSv1)
animals = pd.read_csv("animals.csv").values
# 配置文件和数据生成
output_dir = "output_files"
os.makedirs(output_dir, exist_ok=True)


# 随机字符串生成
def random_string(length=10):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


# 写入大文件
big_file_path = os.path.join(output_dir, "big_file.txt")
with open(big_file_path, "w") as f:
    for i in range(10000):
        line = f"Line {i}: {random_string(50)}"
        f.write(line)

# 读取文件并统计长度
char_count = 0
line_count = 0
with open(big_file_path, "r") as f:
    for line in f:
        char_count += len(line)
        line_count += 1

print(f"Generated file has {line_count} lines and {char_count} characters.")

# 随机数操作
random_numbers = [random.randint(1, 1000) for _ in range(10000)]

# 排序
start_time = time.time()
sorted_numbers = sorted(random_numbers)
end_time = time.time()
print(f"Sorted 10000 numbers in {end_time - start_time:.2f} seconds.")

# 写入排序结果
sorted_file_path = os.path.join(output_dir, "sorted_numbers.json")
with open(sorted_file_path, "w") as f:
    json.dump(sorted_numbers, f)

# 简单搜索
search_target = random_numbers[5000]
index = -1
for i, num in enumerate(sorted_numbers):
    if num == search_target:
        index = i
        break

if index != -1:
    print(f"Found number {search_target} at index {index}.")
else:
    print(f"Number {search_target} not found.")

# 简单统计
num_sum = sum(random_numbers)
average = num_sum / len(random_numbers)
print(f"Sum: {num_sum}, Average: {average:.2f}")


# 文件拆分
def split_file(file_path, lines_per_file):
    with open(file_path, "r") as f:
        file_index = 1
        current_file = open(os.path.join(output_dir, f"split_{file_index}.txt"), "w")
        for i, line in enumerate(f, start=1):
            current_file.write(line)
            if i % lines_per_file == 0:
                current_file.close()
                file_index += 1
                current_file = open(os.path.join(output_dir, f"split_{file_index}.txt"), "w")
        current_file.close()


split_file(big_file_path, 2000)

# 数据生成和分析
big_list = [random.randint(1, 1000) for _ in range(100000)]
even_numbers = [x for x in big_list if x % 2 == 0]
odd_numbers = [x for x in big_list if x % 2 != 0]

print(f"Generated 100000 numbers with {len(even_numbers)} evens and {len(odd_numbers)} odds.")

# 保存数据
big_data_file = os.path.join(output_dir, "big_data.json")
with open(big_data_file, "w") as f:
    json.dump({"evens": even_numbers, "odds": odd_numbers}, f)

# 读回数据并统计
with open(big_data_file, "r") as f:
    data = json.load(f)

print(f"Read back {len(data['evens'])} evens and {len(data['odds'])} odds from file.")

# 简单矩阵运算
matrix_size = 100
matrix = [[random.randint(1, 10) for _ in range(matrix_size)] for _ in range(matrix_size)]

start_time = time.time()
matrix_sum = sum(sum(row) for row in matrix)
end_time = time.time()

print(f"Sum of {matrix_size}x{matrix_size} matrix is {matrix_sum} (calculated in {end_time - start_time:.2f} seconds).")

# 矩阵转置
transposed = [[matrix[j][i] for j in range(matrix_size)] for i in range(matrix_size)]
print("Transposed matrix calculated.")

# 保存矩阵到文件
matrix_file = os.path.join(output_dir, "matrix.txt")
with open(matrix_file, "w") as f:
    for row in transposed:
        f.write(' '.join(map(str, row)))

# 模拟大计算任务
large_sum = 0
for i in range(1, 100000):
    large_sum += i * random.randint(1, 100)

print(f"Large sum calculated: {large_sum}")


# 字符统计
def char_statistics(file_path):
    stats = {}
    with open(file_path, "r") as f:
        for line in f:
            for char in line:
                stats[char] = stats.get(char, 0) + 1
    return stats


stats = char_statistics(big_file_path)
print(f"Character statistics: {len(stats)} unique characters.")

# 总结
summary_file = os.path.join(output_dir, "summary.txt")
with open(summary_file, "w") as f:
    f.write(f"File line count: {line_count}")
    f.write(f"File char count: {char_count}")
    f.write(f"Sorted file: {sorted_file_path}")
    f.write(f"Even numbers: {len(even_numbers)}")
    f.write(f"Odd numbers: {len(odd_numbers)}")
    f.write(f"Matrix sum: {matrix_sum}")
    f.write(f"Large sum: {large_sum}")

print("All tasks completed and results summarized.")

# 配置文件和数据生成
output_dir = "output_files"
os.makedirs(output_dir, exist_ok=True)


# 随机字符串生成
def random_string(length=10):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


# 写入大文件
big_file_path = os.path.join(output_dir, "big_file.txt")
with open(big_file_path, "w") as f:
    for i in range(10000):
        line = f"Line {i}: {random_string(50)}"
        f.write(line)

# 读取文件并统计长度
char_count = 0
line_count = 0
with open(big_file_path, "r") as f:
    for line in f:
        char_count += len(line)
        line_count += 1

print(f"Generated file has {line_count} lines and {char_count} characters.")

# 随机数操作
random_numbers = [random.randint(1, 1000) for _ in range(10000)]

# 排序
start_time = time.time()
sorted_numbers = sorted(random_numbers)
end_time = time.time()
print(f"Sorted 10000 numbers in {end_time - start_time:.2f} seconds.")

# 写入排序结果
sorted_file_path = os.path.join(output_dir, "sorted_numbers.json")
with open(sorted_file_path, "w") as f:
    json.dump(sorted_numbers, f)

# 简单搜索
search_target = random_numbers[5000]
index = -1
for i, num in enumerate(sorted_numbers):
    if num == search_target:
        index = i
        break

if index != -1:
    print(f"Found number {search_target} at index {index}.")
else:
    print(f"Number {search_target} not found.")

# 简单统计
num_sum = sum(random_numbers)
average = num_sum / len(random_numbers)
print(f"Sum: {num_sum}, Average: {average:.2f}")


# 文件拆分
def split_file(file_path, lines_per_file):
    with open(file_path, "r") as f:
        file_index = 1
        current_file = open(os.path.join(output_dir, f"split_{file_index}.txt"), "w")
        for i, line in enumerate(f, start=1):
            current_file.write(line)
            if i % lines_per_file == 0:
                current_file.close()
                file_index += 1
                current_file = open(os.path.join(output_dir, f"split_{file_index}.txt"), "w")
        current_file.close()


split_file(big_file_path, 2000)

# 数据生成和分析
big_list = [random.randint(1, 1000) for _ in range(100000)]
even_numbers = [x for x in big_list if x % 2 == 0]
odd_numbers = [x for x in big_list if x % 2 != 0]

print(f"Generated 100000 numbers with {len(even_numbers)} evens and {len(odd_numbers)} odds.")

# 保存数据
big_data_file = os.path.join(output_dir, "big_data.json")
with open(big_data_file, "w") as f:
    json.dump({"evens": even_numbers, "odds": odd_numbers}, f)

# 读回数据并统计
with open(big_data_file, "r") as f:
    data = json.load(f)

print(f"Read back {len(data['evens'])} evens and {len(data['odds'])} odds from file.")

# 简单矩阵运算
matrix_size = 100
matrix = [[random.randint(1, 10) for _ in range(matrix_size)] for _ in range(matrix_size)]

start_time = time.time()
matrix_sum = sum(sum(row) for row in matrix)
end_time = time.time()

print(f"Sum of {matrix_size}x{matrix_size} matrix is {matrix_sum} (calculated in {end_time - start_time:.2f} seconds).")

# 矩阵转置
transposed = [[matrix[j][i] for j in range(matrix_size)] for i in range(matrix_size)]
print("Transposed matrix calculated.")

# 保存矩阵到文件
matrix_file = os.path.join(output_dir, "matrix.txt")
with open(matrix_file, "w") as f:
    for row in transposed:
        f.write(' '.join(map(str, row)))

# 模拟大计算任务
large_sum = 0
for i in range(1, 100000):
    large_sum += i * random.randint(1, 100)

print(f"Large sum calculated: {large_sum}")


# 字符统计
def char_statistics(file_path):
    stats = {}
    with open(file_path, "r") as f:
        for line in f:
            for char in line:
                stats[char] = stats.get(char, 0) + 1
    return stats


stats = char_statistics(big_file_path)
print(f"Character statistics: {len(stats)} unique characters.")

# 数组与文件操作
large_array = [random.randint(1, 500) for _ in range(500000)]
output_array_file = os.path.join(output_dir, "large_array.txt")

with open(output_array_file, "w") as f:
    for num in large_array:
        f.write(f"{num}")

with open(output_array_file, "r") as f:
    read_array = [int(line.strip()) for line in f]

print(f"Array file created and read back. Size: {len(read_array)}")

# 图生成
node_count = 500
edges = [(random.randint(1, node_count), random.randint(1, node_count)) for _ in range(1000)]

graph_file = os.path.join(output_dir, "graph.txt")
with open(graph_file, "w") as f:
    for edge in edges:
        f.write(f"{edge[0]} {edge[1]}")

print(f"Graph with {len(edges)} edges generated and saved.")


# 数据频率统计
def frequency_analysis(data):
    counter = Counter(data)
    return counter


frequency_file = os.path.join(output_dir, "frequency.txt")
data_frequency = frequency_analysis(large_array)
with open(frequency_file, "w") as f:
    for key, value in data_frequency.items():
        f.write(f"{key}: {value}")

print("Frequency analysis completed and saved.")
import numpy as np


# 数据归一化
def normalize_data(data):
    max_value = max(data)
    min_value = min(data)
    return [(x - min_value) / (max_value - min_value) for x in data]


normalized_data = normalize_data(random_numbers)
normalized_file = os.path.join(output_dir, "normalized_data.json")
with open(normalized_file, "w") as f:
    json.dump(normalized_data, f)

print("Normalized data saved.")

mat2 = np.mat("1 2 3;4 5 6;7 8 9")
print(mat2)
print(type(mat2))
# 利用分块矩阵来构造大的矩阵
arr1 = np.random.randint(1, 10, (2, 3))
arr2 = np.random.randint(1, 10, (2, 3))
arr3 = np.random.randint(1, 10, (2, 3))
arr4 = np.random.randint(1, 10, (2, 3))
mat3 = np.bmat('arr1 arr2;arr3 arr4')
print(mat3)
# 矩阵的运算：加法，减法（矩阵shape的大小一致）,乘法(c=a*b,要求a的列数等于b的行数，c的行数等于a的行数，c的 =列数等于b的列数)，点运算
matr1 = np.mat(np.random.randint(1, 10, (2, 3)))
matr2 = np.mat(np.random.randint(1, 10, (2, 3)))
print(matr1 - matr2)
matr3 = np.mat(np.random.randint(1, 10, (3, 2)))
print(matr1 * matr3)
print('*********')
print(matr1)
print(np.multiply(matr1, matr2))
arr1 = np.random.randint(1, 10, (2, 3))
print(arr2)
print(arr1 ** arr2)
print(np.any(arr1 == arr2))
money = 10000
sunny_water = 3
sunny_food = 4
hot_water = 9
hot_food = 9

# 前四天的天气
weather = 'sunny', 'hot', 'sunny', 'sunny'
# 单人 晴朗 行走 消耗的水与食物以及钱
def single_sunny():
    move_expend_water = 2 * sunny_water
    move_expend_food = 2 * sunny_food
    # 晴朗 水与食物的花费
    cost_money = 5 * move_expend_water + 10 * move_expend_food
    # print('单人 晴朗 行走 消耗的钱:', cost_money)
    return cost_money


# 单人 高温 行走 消耗的水与食物以及钱
def single_hot():
    move_expend_water = 2 * hot_water
    move_expend_food = 2 * hot_food
    # 晴朗 水与食物的花费
    cost_money = 5 * move_expend_water + 10 * move_expend_food
    # print('单人 高温 行走 消耗的钱:', cost_money)
    return cost_money

def double_sunny():
    move_expend_water = 4 * sunny_water
    move_expend_food = 4 * sunny_food
    # 晴朗 水与食物的花费
    cost_money = 5 * move_expend_water + 10 * move_expend_food
    # print('双人 晴朗 行走 消耗的钱:', cost_money)
    return cost_money

# 双人 高温 行走 消耗的水与食物以及钱
def double_hot():
    move_expend_water = 4 * hot_water
    move_expend_food = 4 * hot_food
    # 晴朗 水与食物的花费
    cost_money = 5 * move_expend_water + 10 * move_expend_food
    # print('双人 高温 行走 消耗的钱:', cost_money)
    return cost_money

# 方案一 1个玩家走5号 1个玩家走4号再6号汇合一起走到13号终点，则消耗的钱一样
one_1 = single_sunny() + single_hot() + double_sunny()
one_2 = single_sunny() + single_hot() + double_sunny()
print(money - one_1)
print(money - one_2)

# 方案二 1个玩家走1-5-6-13线，2个玩家走1-4-7-12-13号线
two_1 = single_sunny() + single_hot() + single_sunny()
print(money - two_1)
two_2 = single_sunny() + single_hot() + single_sunny() + single_sunny()
print(money - two_2)

# Experiment 2

def fib_series(n):
    res = []
    if n == 1:
        res.append()
    elif n == 2:
        res.append()
    else:
        res = []
        for i in range(2, n):
            res.append(res[i - 1] + res[i - 2])
    return res

def flib(n):
    if n == 1 or n == 2:
        res = 1
    else:
        res = flib(n - 1) + flib(n - 2)
    return res


# print([flib(i) for i in range(1,6)])
###3
def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n = n + 1


# fab(5)
###4
def fab_1(max):
    n, a, b = 0, 0, 1
    L = []
    while n < max:
        L.append(b)
        a, b = b, a + b
        n = n + 1
    return L

# Experiment 3
def is_prime(x):
    num = []
    i = 2
    for i in range(2, x):
        j = 2
        for j in range(2, i):
            if (i % j == 0):
                break
        else:
            num.append(i)
    return num

def is_prime(x):
    for i in range(2, int(pow(x, 0.5)) + 1):
        if x % i == 0:
            return False
    else:
        return True


def is_prime(n):
    if n <= 1:
        return False
    else:
        for i in range(2, n):
            if n % i == 0:
                return False
        return True


def twin_primes(n):
    res = []  # 用元组的方式保存，保存一对孪生数
    for i in range(2, n + 1):
        if is_prime(i):
            if is_prime(i + 2):
                num = (i, i + 2)
                res.append(num)
    return res


print(twin_primes(100))

import numpy as np

a = np.array([1, 2, 3])
b = np.array(range(10), dtype='float')
c = np.arange(1, 10, 3)
d = np.array(range(10), dtype='bool')
a = a.astype('f8')
print(a.dtype)
print(a * 3)
n3 = np.zeros((4, 3))
print(n3)
print('{}'.format(n3.shape[0]))
print('{}'.format(n3.shape[1]))
n4 = np.arange(1, 13, )
n5 = n4.reshape((3, 4))
print(n5)
print(n5.flatten())
print(n5.reshape(12, 1))
# 向量的计算
a = np.arange(5)
b = np.arange(6, 11)
print(a + b)
print(a * b)
a = np.arange(12).reshape(3, 4)
b = np.arange(1, 13).reshape(3, 4)
print(a)
print('**' * 10)
print(b)
print('**' * 10)
print(a + b)


def mymean(x):
    n, m = x.shape()
    res = []
    for j in range(m):
        res = []
        temp = 0
        k = 0
        for i in range(n):
            if x[i, j] == x[j, i]:
                k = k = 1
                temp += x[i, j]
                res.append(temp / k)
        return res


a = np.arange(8).reshape(2, 4)
b = np.arange(1, 9).reshape(2, 4)
b1 = b[0, 2, :].reshape(1, 4)
c = np.vstack((a, b1))  # 横着拼接
print(c)
print('*' * 20)
d = np.hstack((a, b))  # 竖着拼接
print(d)
a = np.zeros((3, 4))
print(a)
b = np.ones((3, 4))
print(b)
c = np.eye(4)
print(c)
np.random.seed(10)
a = np.random.randint(5, 10)  # 从这个范围内随机选取一个数
print(a)
b = np.random.choice(70, 5)  # 随机从70中选取5数
print(b)
import pandas as pd

# 带索引值的向量，index指定索引值，默认为range（len（list))
# 通过列表的方式
a = pd.Series([1, 2, 3, 4], index=list('abcd'))
print(a)
# 通过字典的方式
dic = {'name': 'Lucy', 'age': 18}
dic = pd.Series(dic)
print(dic)
c = pd.Series(range(5), index=['hh{}'.format(i) for i in range(5)])
print(c)
print(c[4])
print(c['hh4'])
print('********')
print(c[::4])
print('********')
print(c[[1, 3]])
print('********')

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='C:/Windows/Fonts/STSONG.TTF', size=10)
import numpy as np

plt.figure(figsize=(10, 5), dpi=100)  # 设置画布
plt.subplot(2, 2, 1)  # 创建子图（n，m，k）把画布分成n行，m列，在第k个子图换图
x = np.linspace(0, 3 * np.pi, 100)
y = np.sin(x)
y1 = np.cos(x)
# plt.plot(x, y, label='sin(x)', c='c', linestyle=':', marker='D', linewidth=2, markersize=2)
# 添加标注，改变画图的颜色  -改变线的形状 -长虚线   ：虚线  -.长短线  改变点的形状 + * < >
plt.plot(x, y1, label='cos(x)', c='b')

plt.title('我的图片', fontproperties=font)  # 添加标题

plt.xlabel('X')  # 横坐标名称
plt.ylabel('Y')

plt.xlim([0, 3 * np.pi])  # 限制x的取值范围
plt.ylim([-1.5, 1.5])

_x = ['{}点'.format(i) for i in range(len(x))]
plt.xticks(x[::6], _x[::6], fontproperties=font, rotation=45)

# plt.text((0, 0), '原点')

plt.legend(loc='best')  # 添加标注的位置

plt.grid(alpha=0.3, color='red')

plt.show()
# plt.savefig('p.png')

# 实验 1
first_name = "chris"
last_name = "Wilson"
full_name = first_name + " " + last_name
print("Hello, " + full_name.title() + "!" + " " * 3 + "Nice to meet you.")
language = "python"
print(language.upper())
print(language)
product_num = "201906C15M"
print(product_num[6])
print(product_num[-1])
print(product_num[4:6])
print(product_num[:4])
print(product_num[-3:-1])
print("m" in product_num)
# 实验 2
name = "Evan"
money = 45.783
number = 10
print("%10s paid $%-6.1f for %d apples." % (name, money, number))
print("{0:*^10} paid ${1:<6.1f} for {2:d} apples".format(name, money, number))
# 实验 3
str = "HOW DO YOU DO THAT".split()
print(" ".join(str[::-1]))
print(str[::-1])
import numpy as np

a = np.arange(0, 1, 0.01).reshape(10, 10)
# print(arr)
b = np.random.randn(10, 10)
# print(arr1)
print('两个数组相加：')
print(np.add(a, b))

print('两个数组相减：')
print(np.subtract(a, b))

print('两个数组相乘：')
print(np.multiply(a, b))

print('两个数组相除：')
print(np.divide(a, b))
arr = np.random.randint(0, 100, (10, 10))
np.savetxt("out.csv", arr, fmt="%d", delimiter=",")
data = np.loadtxt("out.csv", delimiter=",")
# 排序
print('排序:')
data_sort = np.sort(data, axis=0)
print(data_sort)
# 去重
print('去重:')
data_unique = np.unique(data)
print(data_unique)
# 求和
print('求和:')
data_sum = np.sum(data, axis=0)
print(data_sum)
# 累计求和

print('累计求和')
data_cumsum = np.cumsum(data, axis=0)
print(data_cumsum)
# 最大值
print('最大值：', np.max(data, axis=0))
print(np.argmax(data, axis=0))

print('最小值：', np.min(data, axis=0))
print(np.argmin(data, axis=0))
print('均值：', np.mean(data, axis=0))
print('方差：', np.var(data, axis=0))
print('标准差：', np.std(data, axis=0))

# 任务类
class Task:
    def __init__(self, title, description, priority, due_date, assigned_to=None):
        self.title = title
        self.description = description
        self.priority = priority  # 优先级：高、中、低
        self.due_date = due_date  # 截止日期
        self.assigned_to = assigned_to  # 分配给谁
        self.completed = False  # 是否完成

    def mark_completed(self):
        """标记任务为已完成"""
        self.completed = True
        print(f"任务 '{self.title}' 已标记为完成。")

    def __str__(self):
        """返回任务信息的字符串表示"""
        status = "已完成" if self.completed else "未完成"
        return (
            f"任务: {self.title}\n"
            f"描述: {self.description}\n"
            f"优先级: {self.priority}\n"
            f"截止日期: {self.due_date}\n"
            f"分配给: {self.assigned_to if self.assigned_to else '未分配'}\n"
            f"状态: {status}\n"
        )

# 任务管理系统类
class TaskManager:
    def __init__(self):
        self.tasks = []  # 存储所有任务

    def add_task(self, title, description, priority, due_date, assigned_to=None):
        """添加新任务"""
        task = Task(title, description, priority, due_date, assigned_to)
        self.tasks.append(task)
        print(f"任务 '{title}' 已添加。")

    def assign_task(self, task_title, assigned_to):
        """分配任务给用户"""
        task = self._find_task(task_title)
        if task:
            task.assigned_to = assigned_to
            print(f"任务 '{task_title}' 已分配给 {assigned_to}。")
        else:
            print(f"任务 '{task_title}' 不存在。")

    def mark_task_completed(self, task_title):
        """标记任务为已完成"""
        task = self._find_task(task_title)
        if task:
            task.mark_completed()
        else:
            print(f"任务 '{task_title}' 不存在。")

    def show_tasks(self, sort_by="priority"):
        """显示所有任务，支持按优先级、截止日期或状态排序"""
        if not self.tasks:
            print("没有任务。")
            return

        # 排序逻辑
        if sort_by == "priority":
            sorted_tasks = sorted(self.tasks, key=lambda x: x.priority)
        elif sort_by == "due_date":
            sorted_tasks = sorted(self.tasks, key=lambda x: datetime.strptime(x.due_date, "%Y-%m-%d"))
        elif sort_by == "status":
            sorted_tasks = sorted(self.tasks, key=lambda x: x.completed)
        else:
            sorted_tasks = self.tasks

        for task in sorted_tasks:
            print(task)

    def search_tasks(self, keyword):
        """根据关键字搜索任务"""
        found_tasks = [task for task in self.tasks if keyword.lower() in task.title.lower() or keyword.lower() in task.description.lower()]
        if not found_tasks:
            print(f"未找到包含 '{keyword}' 的任务。")
            return
        for task in found_tasks:
            print(task)

    def save_to_file(self, filename="tasks.json"):
        """保存任务数据到文件"""
        data = []
        for task in self.tasks:
            task_data = {
                "title": task.title,
                "description": task.description,
                "priority": task.priority,
                "due_date": task.due_date,
                "assigned_to": task.assigned_to,
                "completed": task.completed
            }
            data.append(task_data)
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"任务数据已保存到文件 {filename}。")

    def load_from_file(self, filename="tasks.json"):
        """从文件加载任务数据"""
        if not os.path.exists(filename):
            print(f"文件 {filename} 不存在。")
            return
        with open(filename, "r") as file:
            data = json.load(file)
        for task_data in data:
            task = Task(
                task_data["title"],
                task_data["description"],
                task_data["priority"],
                task_data["due_date"],
                task_data["assigned_to"]
            )
            task.completed = task_data["completed"]
            self.tasks.append(task)
        print(f"任务数据已从文件 {filename} 加载。")

    def _find_task(self, task_title):
        """根据任务名称查找任务"""
        for task in self.tasks:
            if task.title.lower() == task_title.lower():
                return task
        return None

# 主菜单
def main_menu():
    manager = TaskManager()
    while True:
        print("\n===== 任务管理系统 =====")
        print("1. 添加任务")
        print("2. 分配任务")
        print("3. 标记任务完成")
        print("4. 查看任务列表")
        print("5. 搜索任务")
        print("6. 保存任务数据")
        print("7. 加载任务数据")
        print("8. 退出")
        choice = input("请选择操作: ")

        if choice == "1":
            title = input("请输入任务名称: ")
            description = input("请输入任务描述: ")
            priority = input("请输入任务优先级（高、中、低）: ")
            due_date = input("请输入截止日期（格式: YYYY-MM-DD）: ")
            manager.add_task(title, description, priority, due_date)

        elif choice == "2":
            task_title = input("请输入任务名称: ")
            assigned_to = input("请输入分配给谁: ")
            manager.assign_task(task_title, assigned_to)

        elif choice == "3":
            task_title = input("请输入任务名称: ")
            manager.mark_task_completed(task_title)

        elif choice == "4":
            sort_by = input("按什么排序？（priority/due_date/status）: ") or "priority"
            manager.show_tasks(sort_by)

        elif choice == "5":
            keyword = input("请输入搜索关键字: ")
            manager.search_tasks(keyword)

        elif choice == "6":
            filename = input("请输入保存文件名（默认: tasks.json）: ") or "tasks.json"
            manager.save_to_file(filename)

        elif choice == "7":
            filename = input("请输入加载文件名（默认: tasks.json）: ") or "tasks.json"
            manager.load_from_file(filename)

        elif choice == "8":
            print("退出系统。")
            break

        else:
            print("无效选择，请重试。")

# 日记数据存储
diary_entries = []

# 添加日记
def add_entry():
    title = input("请输入日记标题: ")
    content = input("请输入日记内容: ")
    date = input("请输入日期（格式: YYYY-MM-DD，留空使用今天）: ") or datetime.now().strftime("%Y-%m-%d")

    entry = {
        "title": title,
        "content": content,
        "date": date
    }
    diary_entries.append(entry)
    print(f"日记 '{title}' 已添加。")


# 查看日记
def view_entries(sort_by="date"):
    if not diary_entries:
        print("没有日记。")
        return

    # 排序逻辑
    if sort_by == "date":
        sorted_entries = sorted(diary_entries, key=lambda x: x["date"])
    else:
        sorted_entries = diary_entries

    for entry in sorted_entries:
        print(f"\n标题: {entry['title']}")
        print(f"日期: {entry['date']}")
        print(f"内容: {entry['content']}")
        print("-" * 30)


# 搜索日记
def search_entries(keyword):
    found_entries = [entry for entry in diary_entries if
                     keyword.lower() in entry["title"].lower() or keyword.lower() in entry["content"].lower()]
    if not found_entries:
        print(f"未找到包含 '{keyword}' 的日记。")
        return
    for entry in found_entries:
        print(f"\n标题: {entry['title']}")
        print(f"日期: {entry['date']}")
        print(f"内容: {entry['content']}")
        print("-" * 30)


# 删除日记
def delete_entry():
    title = input("请输入要删除的日记标题: ")
    global diary_entries
    initial_length = len(diary_entries)
    diary_entries = [entry for entry in diary_entries if entry["title"].lower() != title.lower()]
    if len(diary_entries) < initial_length:
        print(f"日记 '{title}' 已删除。")
    else:
        print(f"未找到标题为 '{title}' 的日记。")


# 保存日记数据到文件
def save_to_file(filename="diary.json"):
    data = diary_entries
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    print(f"日记数据已保存到文件 {filename}。")


# 从文件加载日记数据
def load_from_file(filename="diary.json"):
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在。")
        return
    with open(filename, "r") as file:
        data = json.load(file)
    global diary_entries
    diary_entries = data
    print(f"日记数据已从文件 {filename} 加载。")


# 主菜单
def main_menu():
    while True:
        print("\n===== 简易日记管理系统 =====")
        print("1. 添加日记")
        print("2. 查看日记")
        print("3. 搜索日记")
        print("4. 删除日记")
        print("5. 保存日记数据")
        print("6. 加载日记数据")
        print("7. 退出")
        choice = input("请选择操作: ")

        if choice == "1":
            add_entry()

        elif choice == "2":
            sort_by = input("按什么排序？（date，留空默认）: ") or "date"
            view_entries(sort_by)

        elif choice == "3":
            keyword = input("请输入搜索关键字: ")
            search_entries(keyword)

        elif choice == "4":
            delete_entry()

        elif choice == "5":
            filename = input("请输入保存文件名（默认: diary.json）: ") or "diary.json"
            save_to_file(filename)

        elif choice == "6":
            filename = input("请输入加载文件名（默认: diary.json）: ") or "diary.json"
            load_from_file(filename)

        elif choice == "7":
            print("退出系统。")
            break

        else:
            print("无效选择，请重试。")


import os
import json

# 通讯录数据存储
contacts = []


# 添加联系人
def add_contact():
    name = input("请输入姓名: ")
    phone = input("请输入电话号码: ")
    email = input("请输入电子邮件: ")

    contact = {
        "name": name,
        "phone": phone,
        "email": email
    }
    contacts.append(contact)
    print(f"联系人 '{name}' 已添加。")


# 查看联系人
def view_contacts(sort_by="name"):
    if not contacts:
        print("没有联系人。")
        return

    # 排序逻辑
    if sort_by == "name":
        sorted_contacts = sorted(contacts, key=lambda x: x["name"])
    elif sort_by == "phone":
        sorted_contacts = sorted(contacts, key=lambda x: x["phone"])
    else:
        sorted_contacts = contacts

    for contact in sorted_contacts:
        print(f"\n姓名: {contact['name']}")
        print(f"电话号码: {contact['phone']}")
        print(f"电子邮件: {contact['email']}")
        print("-" * 30)


# 搜索联系人
def search_contacts(keyword):
    found_contacts = [contact for contact in contacts if
                      keyword.lower() in contact["name"].lower() or keyword.lower() in contact["phone"].lower()]
    if not found_contacts:
        print(f"未找到包含 '{keyword}' 的联系人。")
        return
    for contact in found_contacts:
        print(f"\n姓名: {contact['name']}")
        print(f"电话号码: {contact['phone']}")
        print(f"电子邮件: {contact['email']}")
        print("-" * 30)


# 删除联系人
def delete_contact():
    name = input("请输入要删除的联系人姓名: ")
    global contacts
    initial_length = len(contacts)
    contacts = [contact for contact in contacts if contact["name"].lower() != name.lower()]
    if len(contacts) < initial_length:
        print(f"联系人 '{name}' 已删除。")
    else:
        print(f"未找到姓名为 '{name}' 的联系人。")


# 保存通讯录数据到文件
def save_to_file(filename="contacts.json"):
    data = contacts
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    print(f"通讯录数据已保存到文件 {filename}。")


# 从文件加载通讯录数据
def load_from_file(filename="contacts.json"):
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在。")
        return
    with open(filename, "r") as file:
        data = json.load(file)
    global contacts
    contacts = data
    print(f"通讯录数据已从文件 {filename} 加载。")


# 主菜单
def main_menu():
    while True:
        print("\n===== 简易通讯录管理系统 =====")
        print("1. 添加联系人")
        print("2. 查看联系人")
        print("3. 搜索联系人")
        print("4. 删除联系人")
        print("5. 保存通讯录数据")
        print("6. 加载通讯录数据")
        print("7. 退出")
        choice = input("请选择操作: ")

        if choice == "1":
            add_contact()

        elif choice == "2":
            sort_by = input("按什么排序？（name/phone，留空默认）: ") or "name"
            view_contacts(sort_by)

        elif choice == "3":
            keyword = input("请输入搜索关键字: ")
            search_contacts(keyword)

        elif choice == "4":
            delete_contact()

        elif choice == "5":
            filename = input("请输入保存文件名（默认: contacts.json）: ") or "contacts.json"
            save_to_file(filename)

        elif choice == "6":
            filename = input("请输入加载文件名（默认: contacts.json）: ") or "contacts.json"
            load_from_file(filename)

        elif choice == "7":
            print("退出系统。")
            break
