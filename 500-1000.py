import os
import random
import string
import json
import time
from collections import Counter

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

# 大计算任务
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


def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return len(lines)


def count_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        words = content.split()  # 按空格分割成单词
        return len(words)


def count_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return len(content)


def analyze_directory(directory_path):
    # 遍历目录中的每个文件
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.endswith('.txt'):  # 仅分析txt文件
                print(f"Analyzing file: {file_name}")
                lines = count_lines(file_path)
                words = count_words(file_path)
                chars = count_characters(file_path)
                print(f"Lines: {lines}, Words: {words}, Characters: {chars}")
                print('-' * 40)


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
print('*****')
print(matr1.T)  # 矩阵的转置
print(matr1.H)  # 矩阵的共轭
print(matr1.I)  # 矩阵的逆
print(matr1.A)  # 返回矩阵的数组视图
print('******')
print(arr1)
print(arr2)
print(arr1 ** arr2)  # 幂次方
print(np.any(arr1 == arr2))  # any只需一个相等，all需要全部相等才返回true
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


##Experiment 2
###1
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


###2
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


a = np.array([1, 2, 3])
b = np.array(range(10), dtype='float')
c = np.arange(1, 10, 3)
d = np.array(range(10), dtype='bool')
a = a.astype('f8')
print(a.dtype)
print(a * 3)
n3 = np.zeros((4, 3))  # 二维
print(n3)
print('{}'.format(n3.shape[0]))
print('{}'.format(n3.shape[1]))
n4 = np.arange(1, 13, )
n5 = n4.reshape((3, 4))
print(n5)
print(n5.flatten())  # 二维转一行
print(n5.reshape(12, 1))  # 转一列
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from calendar import month_name
from matplotlib.ticker import AutoMinorLocator

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
# 产生x和y的数据
x = np.arange(1, 13, 1)
y = 2 * x
ax.plot(x, y, ls="-", lw=2, color="orange", marker="o", ms=20, mfc="c", mec="c")
# 设置内层刻度为主刻度的四等分
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
# 设置主刻度为1月12月
plt.xticks(x, month_name[1:13], rotation=25)
# 设置X轴和Y轴的最小值和最大值
ax.set_xlim(0, 13)
ax.set_ylim(0, 30)
plt.show()
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
