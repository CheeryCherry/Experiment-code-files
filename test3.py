import os
import json
import time
import ssl
import pickle
import sys
import pandas as pd

output_dir = "output_files"
os.makedirs(output_dir, exist_ok=True)


def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)


print(factorial(5))  # 输出: 120


def fibonacci(limit):
    a, b = 0, 1
    while a < limit:
        print(a, end=' ')
        a, b = b, a + b


fibonacci(100)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 过滤偶数
evens = [x for x in numbers if x % 2 == 0]

# 平方每个数字
squares = list(map(lambda x: x ** 2, numbers))

print(evens)  # 输出: [2, 4, 6, 8]
print(squares)  # 输出: [1, 4, 9, 16, 25, 36, 49, 64, 81]


def write_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)


def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()


write_to_file('example.txt', 'Hello, World!')
print(read_file('example.txt'))  # 输出: Hello, World!


def count_words(text):
    words = text.lower().split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count


text = "Hello world hello python world"
print(count_words(text))  # 输出: {'hello': 2, 'world': 2, 'python': 1}


def celsius_to_fahrenheit(c):
    return (c * 9 / 5) + 32


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9


print(f"32°F = {fahrenheit_to_celsius(32):.1f}°C")  # 输出: 32°F = 0.0°C
print(f"100°C = {celsius_to_fahrenheit(100):.1f}°F")  # 输出: 100°C = 212.0°F


def calculate(a, b, operator):
    operations = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y if y != 0 else "不能除以零"
    }
    return operations.get(operator, lambda x, y: "无效运算符")(a, b)


print(calculate(10, 5, '+'))  # 输出: 15
print(calculate(10, 0, '/'))  # 输出: 不能除以零


def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]


print(is_palindrome("A man, a plan, a canal: Panama"))  # 输出: True
print(is_palindrome("Hello World"))  # 输出: False


def find_longest_word(filename):
    with open(filename, 'r') as f:
        words = f.read().split()
        return max(words, key=len) if words else None


# 有一个test.txt文件
longest = find_longest_word('test.txt')
print(f"最长的单词是: {longest}") if longest else print("文件为空")
with open('a.pkl', 'rb') as file:
    data = pickle.load(file)


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

# 大计算任务
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


def remove_duplicates(items):
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]


numbers = [3, 5, 2, 3, 8, 5, 9, 2]
print(remove_duplicates(numbers))  # 输出: [3, 5, 2, 8, 9]


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


nested = [[1, 2], [3, 4, 5], [6]]
print(flatten(nested))  # 输出: [1, 2, 3, 4, 5, 6]


def batch_rename_files(directory, prefix):
    for i, filename in enumerate(os.listdir(directory)):
        ext = os.path.splitext(filename)[1]
        new_name = f"{prefix}_{i + 1}{ext}"
        os.rename(
            os.path.join(directory, filename),
            os.path.join(directory, new_name)
        )
        print(f"重命名 {filename} 为 {new_name}")


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def primes_up_to(limit):
    return [n for n in range(2, limit + 1) if is_prime(n)]


print(primes_up_to(50))  # 输出50以内的素数

for i in range(1, 5):
    for j in range(1, 5):
        for k in range(1, 5):
            if (i != k) and (i != j) and (j != k):
                print(i, j, k)

i = int(input('净利润:'))
arr = [1000000, 600000, 400000, 200000, 100000, 0]
rat = [0.01, 0.015, 0.03, 0.05, 0.075, 0.1]
r = 0


# 定义求解函数
def find_integer():
    for i in range(2, 85):  # i 的范围从 2 到 84
        if 168 % i == 0:  # i 是 168 的因子
            j = 168 // i  # 计算对应的 j
            if i > j and (i + j) % 2 == 0 and (i - j) % 2 == 0:
                m = (i + j) // 2
                n = (i - j) // 2
                x = n * n - 100
                print(f"x: {x}, m: {m}, n: {n}")


# 调用求解函数
find_integer()

year = int(input('year:\n'))
month = int(input('month:\n'))
day = int(input('day:\n'))

months = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
if 0 < month <= 12:
    sum = months[month - 1]
else:
    print('data error')

leap = 0
if (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
    leap = 1

print('it is the %dth day.' % sum)
a = [1, 2, 3]
b = a[:]
print(b)
for i in range(1, 10):
    print()
    for j in range(1, i + 1):
        print("%d*%d=%d" % (i, j, i * j), end=" ")
f1 = 1
f2 = 1
for i in range(1, 22):
    print('%12ld %12ld' % (f1, f2), end=" ")
    if (i % 3) == 0:
        print('')
    f1 = f1 + f2
    f2 = f1 + f2


def reduceNum(n):
    print('{} = '.format(n), end=" ")
    if not isinstance(n, int) or n <= 0:
        print('请输入一个正确的数字 !')
        exit(0)
    elif n in [1]:
        print('{}'.format(n))
    while n not in [1]:  # 循环保证递归
        for index in range(2, n + 1):
            if n % index == 0:
                n //= index  # n 等于 n//index
                if n == 1:
                    print(index)
                else:  # index 一定是素数
                    print('{} *'.format(index), end=" ")
                break


reduceNum(90)
reduceNum(100)
score = int(input('输入分数:\n'))
if score >= 90:
    grade = 'A'
elif score >= 60:
    grade = 'B'
else:
    grade = 'C'

print('%d 属于 %s' % (score, grade))

s = input('请输入一个字符串:\n')
letters = 0
space = 0
digit = 0
others = 0
for c in s:
    if c.isalpha():
        letters += 1
    elif c.isspace():
        space += 1
    elif c.isdigit():
        digit += 1
    else:
        others += 1
print(letters, space, digit, others)

x2 = 1
for day in range(9, 0, -1):
    x1 = (x2 + 1) * 2
    x2 = x1


def print_diamond(rows):
    # 上半部分
    for i in range(1, rows, 2):
        spaces = " " * ((rows - i) // 2)
        stars = "*" * i
        print(spaces + stars)

    # 下半部分
    for i in range(rows, 0, -2):
        spaces = " " * ((rows - i) // 2)
        stars = "*" * i
        print(spaces + stars)


# 设置行数，可以根据需要调整
rows = 7
print_diamond(rows)

aa = 2
bb = 1
s1 = 0
for n in range(1, 21):
    print(n)
print(s1)

s2 = 0
for n in range(1, 21):
    print(n)
print(s2)
s3 = 0
l1 = range(1, 21)
try:
    pass
except Exception:
    pass


def op(x):
    r = 1
    for i in range(1, x + 1):
        r *= i
    return r


def output(s, l):
    if l == 0:
        return
    print(s[l - 1])
    output(s, l - 1)


s = input('Input a string:')
l = len(s)
output(s, l)

x = int(input("请输入一个数:\n"))
a2 = x // 10000
b2 = x % 10000 // 1000
c2 = x % 1000 // 100
d2 = x % 100 // 10
e2 = x % 10

if a2 != 0:
    print("5 位数：", e2, d2, c2, b2, a2)
elif b2 != 0:
    print("4 位数：", e2, d2, c2, b2)
elif c2 != 0:
    print("3 位数：", e2, d2, c2)
elif d2 != 0:
    print("2 位数：", e2, d2)
else:
    print("1 位数：", e2)

letter = input("please input:")
# while letter  != 'Y':
if letter == 'S':
    print('please input second letter:')
    letter = input("please input:")
    if letter == 'a':
        print('Saturday')
    elif letter == 'u':
        print('Sunday')
    else:
        print('data error')

elif letter == 'F':
    print('Friday')

elif letter == 'M':
    print('Monday')

elif letter == 'T':
    print('please input second letter')
    letter = input("please input:")

    if letter == 'u':
        print('Tuesday')
    elif letter == 'h':
        print('Thursday')
    else:
        print('data error')

elif letter == 'W':
    print('Wednesday')
else:
    print('data error')

al = [1, 4, 6, 9, 13, 16, 19, 28, 40, 100, 0]
print('原始列表:')
for i in range(len(al)):
    print(al[i])
number = int(input("\n插入一个数字:\n"))
end = al[9]
if number > end:
    al[10] = number
else:
    for i in range(10):
        if al[i] > number:
            temp1 = al[i]
            al[i] = number
            for j in range(i + 1, 11):
                temp2 = al[j]
                al[j] = temp1
                temp1 = temp2
            break
print('排序后列表:')
for i in range(11):
    print(al[i])

tmp1 = 0
for i in range(1,101):
    tmp1 += i
print ('The sum is %d' % tmp1)

TRUE = 1
FALSE = 0
def SQ(x):
    return x * x
print ('如果输入的数字小于 50，程序将停止运行。')
again = 1
while again:
    num = int(input('请输入一个数字：'))
    print ('运算结果为: %d' % (SQ(num)))
    if SQ(num) >= 50:
        again = TRUE
    else:
        again = FALSE
