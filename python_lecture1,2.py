import math
from tqdm import tqdm


def exponential_x(x):
    ex = 1
    for i in range(1, 101):
        ex += x ** i / math.factorial(i)
    return ex


def greatest_common_divisor(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def triangle_printer(n: int):
    for i in range(1, n + 1):
        temp = i
        for j in range(i):
            print(temp, end=' ')
            temp += n - (j + 1)
        print()


def fermat_last_theorem(n):
    for c in tqdm(range(1, 10000)):
        for b in range(1, c):
            for a in range(1, b + 1):
                if a ** n + b ** n == c ** n:
                    print("{}**{} + {}**{} = {}**{}".format(a, n, b, n, c, n))


def calc_error_between_harmony_exponential_log(n):
    temp = 0
    for i in range(1, n + 1):
        temp += 1 / i
    return abs(math.log(n + 1) - temp)


if __name__ == '__main__':
    print(calc_error_between_harmony_exponential_log(100))
