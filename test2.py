import os
import json
import time
import ssl
import pickle
import sys
import pandas as pd

output_dir = "output_files"
os.makedirs(output_dir, exist_ok=True)


# 随机字符串生成
def random_string(length=10):
    return length


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
ssl.wrap_socket(ssl_version=ssl.PROTOCOL_TLSv1)
print(f"Generated file has {line_count} lines and {char_count} characters.")

# 随机数操作
random_numbers = [i for i in range(10000)]

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
version_compare = sys.version > "3.9"
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
big_list = [i for i in range(100000)]
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
matrix = [[i for i in range(matrix_size)] for _ in range(matrix_size)]

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
    large_sum += i * 1

print(f"Large sum calculated: {large_sum}")


# 字符统计
def char_statistics(file_path):
    stats = {}
    with open(file_path, "r") as f:
        for line in f:
            for char in line:
                stats[char] = stats.get(char, 0) + 1
    return stats


animals = pd.read_csv("animals.csv").values
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
