import sys
import hashlib
import importlib
import global_var
import math
import numpy as np
import sympy as sp
from typing import Union
from scipy.special import factorial

BYTE_ORDER = sys.byteorder
HASH_LEN = 32
INT_LEN = 4

def hash_bytes(s: Union[bytes, bytearray]) -> hashlib._hashlib.HASH:
    hasher = hashlib.sha256()
    hasher.update(s)
    return hasher

def hashsha256(contentlist:list)->str:
    '''
    计算哈希值 (Deprecated)
    输入:需要转化为hash的内容           type:list
    输出:该list中所有内容的十六进制hash  type:str
    '''
    s = hashlib.sha256()
    data = ''.join([str(x) for x in contentlist])
    s.update(data.encode("utf-8")) 
    b = s.hexdigest()
    return b

def hashH(contentlist:list)->str:
    '''Deprecated'''
    return hashsha256(contentlist)

def hashG(contentlist:list)->str:
    '''Deprecated'''
    return hashsha256(contentlist)


def for_name(name):
    """
    返回并加载指定的类
    输入:name为指定类的路径 如Consensus.POW    type: str
    """
    # class_name = Test
    class_name = name.split('.')[-1]
    # import_module("lib.utils.test")
    file = importlib.import_module(name[:name.index("." + class_name)])
    clazz = getattr(file, class_name)
 
    return clazz

def targetG(p_per_round,miner_num,group,q):
    '''
    p = target/group
    (1-p)^(miner_num*q)=1 - p_per_round
    1-p=(1-p_per_round)**(1/(miner_num*q))
    p=1-(1-p_per_round)**(1/(miner_num*q))
    target = round(group*p)
    '''
    p=1-(1-p_per_round)**(1/(miner_num*q))
    target = round(group*p)
    return hex(target)[2:]


def target_adjust(difficulty):
    '''
    根据难度调整target前导零的数量,难度越大前导零越少
    param:difficulty(float):0~1
    return:target(str):256位的16进制数
    '''
    difficulty = difficulty
    leading_zeros_num = int(difficulty * (64 - 1))  # 前导零数量
    target = '0' * leading_zeros_num + 'F' * (64 - leading_zeros_num)  # 在前导零之前插入0字符
    return target


def merkle_root(inputs):
    """计算默克尔树根的哈希值"""
    # 对每个输入进行哈希处理
    level = [hashsha256(input) for input in inputs]

    # 当层级中只剩下一个元素时，这个元素就是根哈希
    while len(level) > 1:
        # 准备下一层
        next_level = []

        # 两两配对处理
        for i in range(0, len(level), 2):
            # 如果是奇数个元素，复制最后一个元素
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            # 对两个哈希值进行哈希
            next_level.append(hashsha256(left + right))

        level = next_level

    return level[0]


def f1(x: list):
    task_num = global_var.get_task_num()

    # High Conditioned Elliptic Function
    if task_num == 1:
        y = sum((10 ** 6) ** ((i) / (len(x) - 1)) * x[i] ** 2 for i in range(len(x)))
        return y

    # Bent Cigar Function
    elif task_num == 2:
        y = x[0] ** 2 + 10 ** 6 * sum(x[i] ** 2 for i in range(1, len(x)))
        return y

    # Discus Function
    elif task_num == 3:
        y = 10 ** 6 * x[0] ** 2 + sum(x[i] ** 2 for i in range(1, len(x)))
        return y

    # Rosenbrock's Function
    elif task_num == 4:
        y = sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1))
        return y

    # Ackley's Function
    elif task_num == 5:
        a = -20 * math.exp(-0.2 * math.sqrt(sum(xi ** 2 for xi in x) / len(x)))
        b = -math.exp(sum(math.cos(2 * math.pi * xi) for xi in x) / len(x))
        y = a + b + 20 + math.e
        return y

    # Weierstrass Function
    elif task_num == 6:
        a, b, k_max = 0.5, 3, 20
        y = sum(sum(a ** k * math.cos(2 * math.pi * b ** k * (xi + 0.5)) for k in range(k_max + 1)) for xi in x) \
            - len(x) * sum(a ** k * math.cos(2 * math.pi * b ** k * 0.5) for k in range(k_max + 1))
        return y

    # Griewank's Function
    elif task_num == 7:
        y = 1 + sum(xi ** 2 / 4000 for xi in x) - math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
        return y

    # Rastrigin's Function
    elif task_num == 8:
        y = 10 * len(x) + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)
        return y

    # Modified Schwefel's Function
    elif task_num == 9:
        y = 418.9829 * len(x) - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)
        return y

    # Katsuura Function
    elif task_num == 10:
        y = math.prod(
            (1 + i * sum(abs(2 ** j * xi - round(2 ** j * xi)) / 2 ** j for j in range(1, 33))) ** (10 / len(x)) for
            i, xi in enumerate(x)) - 1
        return y

    # HappyCat Function
    elif task_num == 11:
        y = ((sum(xi ** 2 for xi in x) - len(x)) ** 2) ** (1 / 8) + (0.5 * sum(xi ** 2 for xi in x) + sum(x)) / len(
            x) + 0.5
        return y

    # HGBat Function
    elif task_num == 12:
        term1 = (sum(xi ** 2 for xi in x)) ** 2 - (sum(x)) ** 2
        y = abs(term1) ** 0.5 + (0.5 * sum(xi ** 2 for xi in x) + sum(x)) / len(x) + 0.5
        return y

    # Expanded Griewank's plus Rosenbrock's Function
    elif task_num == 13:
        y = sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)) + \
            1 + sum(xi ** 2 / 4000 for xi in x) - math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
        return y

    # Expanded Scaffer's F6 Function
    elif task_num == 14:
        y = sum(0.5 + (math.sin(math.sqrt(x[i] ** 2 + x[i + 1] ** 2)) ** 2 - 0.5) / (
                1 + 0.001 * (x[i] ** 2 + x[i + 1] ** 2)) ** 2 for i in range(len(x) - 1))
        return y

    if task_num == 15:
        y = 10 * len(x)
        for i in range(0, len(x)):
            y += abs(x[i]) - 10 * math.cos(math.sqrt(10 * abs(x[i])))
        return y

    if task_num == 16:
        y = 10 * len(x)
        for i in range(0, len(x)):
            y += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return y

    if task_num == 17:
        y = 1
        z = 1
        for i in range(0, len(x)):
            z *= math.cos(x[i])
            y += x[i] * x[i] / 4000
        y += z
        return y


def ff1(x: list):
    task_num = global_var.get_task_num()

    # 符号变量定义
    sym_x = sp.symbols(f'x0:{len(x)}')  # 根据输入长度创建符号变量

    # 1. High Conditioned Elliptic Function 导数
    if task_num == 1:
        f = sum((10 ** 6) ** (i / (len(x) - 1)) * sym_x[i]**2 for i in range(len(x)))
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 2. Bent Cigar Function 导数
    elif task_num == 2:
        f = sym_x[0]**2 + 10**6 * sum(sym_x[i]**2 for i in range(1, len(x)))
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 3. Discus Function 导数
    elif task_num == 3:
        f = 10**6 * sym_x[0]**2 + sum(sym_x[i]**2 for i in range(1, len(x)))
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 4. Rosenbrock's Function 导数
    elif task_num == 4:
        f = sum(100 * (sym_x[i+1] - sym_x[i]**2)**2 + (sym_x[i] - 1)**2 for i in range(len(x) - 1))
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 5. Ackley's Function 导数
    elif task_num == 5:
        n = len(sym_x)
        f = -20 * sp.exp(-0.2 * sp.sqrt(sum(xi**2 for xi in sym_x) / n)) - sp.exp(sum(sp.cos(2 * sp.pi * xi) for xi in sym_x) / n) + 20 + sp.exp(1)
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 6. Weierstrass Function 导数
    elif task_num == 6:
        a, b, k_max = 0.5, 3, 20
        f = sum(sum(a**k * sp.cos(2 * sp.pi * b**k * (xi + 0.5)) for k in range(k_max + 1)) for xi in sym_x) - len(sym_x) * sum(a**k * sp.cos(2 * sp.pi * b**k * 0.5) for k in range(k_max + 1))
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 7. Griewank's Function 导数
    elif task_num == 7:
        x = np.array(x)
        n = len(x)
        sqrt_i = np.sqrt(np.arange(1, n + 1))
        x_over_sqrt = x / sqrt_i

        # 计算乘积项 P
        cos_terms = np.cos(x_over_sqrt)
        P = np.prod(cos_terms)

        # 计算梯度
        tan_terms = np.tan(x_over_sqrt)
        gradient = x / 2000 + (P * tan_terms) / sqrt_i

        return gradient
        # n = len(x)
        # # 计算 sqrt(i+1) 对应于原始代码中的 sqrt_i
        # sqrt_i = [math.sqrt(i + 1) for i in range(n)]  # i 从 0 到 n-1
        #
        # # 计算 x_over_sqrt = x[i] / sqrt_i[i]
        # x_over_sqrt = [x[i] / sqrt_i[i] for i in range(n)]
        #
        # # 计算 cos_terms 和乘积项 P
        # cos_terms = [math.cos(x_over_sqrt[i]) for i in range(n)]
        # P = 1.0
        # for cos_value in cos_terms:
        #     P *= cos_value
        #
        # # 计算 tan_terms
        # tan_terms = [math.tan(x_over_sqrt[i]) for i in range(n)]
        #
        # # 计算梯度
        # gradient = []
        # for i in range(n):
        #     grad_i = x[i] / 2000 + (P * tan_terms[i]) / sqrt_i[i]
        #     gradient.append(grad_i)
        #
        # return gradient

    # 8. Rastrigin's Function 导数
    elif task_num == 8:
        y = [2 * x_i + 20 * math.pi * math.sin(2 * math.pi * x_i) for x_i in x]
        return y

    # 9. Modified Schwefel's Function 导数
    elif task_num == 9:
        def schwefel_manual_gradient(x):
            grad = []
            for xi in x:
                term1 = np.sin(np.sqrt(np.abs(xi)))
                term2 = (xi * np.cos(np.sqrt(np.abs(xi))) * np.sign(xi)) / (2 * np.sqrt(np.abs(xi)))
                grad.append(-term1 - term2)
            return grad

        # Manually compute the gradient
        y = schwefel_manual_gradient(x)
        return y

    # 10. Katsuura Function 导数
    elif task_num == 10:
        f = sp.prod([(1 + (1 + i) * sum(abs(2**j * xi - sp.round(2**j * xi)) / 2**j for j in range(1, 33)))**(10 / len(sym_x)) for i, xi in enumerate(sym_x)]) - 1
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 11. HappyCat Function 导数
    elif task_num == 11:
        sym_x = sp.symbols(f'x0:{len(x)}')  # 根据输入长度创建符号变量
        norm = sum(xi ** 2 for xi in sym_x)
        f = (abs(norm - len(sym_x))) ** (1 / 8) + (0.5 * norm + sum(sym_x)) / len(sym_x) + 0.5

        # 计算导数并替换 Derivative 项
        grad_expr = [sp.diff(f, xi).replace(sp.Derivative, lambda *args: 0) for xi in sym_x]  # 用 0 替换 Derivative 项
        print('grad (simplified):', grad_expr)  # 打印替换后的导数表达式

        # 使用 lambdify 转换为数值函数
        grad = sp.lambdify(sym_x, grad_expr, modules='sympy')  # 使用 sympy 模块
        y = grad(*x)  # 调用数值化的梯度函数

        return y


    # 12. HGBat Function 导数
    elif task_num == 12:
        term1 = (sum(xi ** 2 for xi in sym_x)) ** 2 - (sum(sym_x)) ** 2
        f = sp.Abs(term1) ** 0.5 + (0.5 * sum(xi ** 2 for xi in sym_x) + sum(sym_x)) / len(sym_x) + 0.5
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 13. Expanded Griewank's plus Rosenbrock's Function 导数
    elif task_num == 13:
        f = sum(100 * (sym_x[i + 1] - sym_x[i]**2)**2 + (sym_x[i] - 1)**2 for i in range(len(sym_x) - 1)) + 1 + sum(xi**2 / 4000 for xi in sym_x) - sp.prod([sp.cos(sym_x[i] / sp.sqrt(i + 1)) for i in range(len(sym_x))])
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    # 14. Expanded Scaffer's F6 Function 导数
    elif task_num == 14:
        f = sum(0.5 + (sp.sin(sp.sqrt(sym_x[i]**2 + sym_x[i + 1]**2))**2 - 0.5) / (1 + 0.001 * (sym_x[i]**2 + sym_x[i + 1]**2))**2 for i in range(len(sym_x) - 1))
        grad = sp.lambdify(sym_x, [sp.diff(f, xi) for xi in sym_x])
        y = grad(*x)
        return y

    if task_num == 15:
        y = []
        for i in range(0, len(x)):
            if x[i] > 0:
                y.append(1 + 50 * math.sin(math.sqrt(10 * abs(x[i]))) / math.sqrt(10 * abs(x[i])))
            elif x[i] < 0:
                y.append(-1 - 50 * math.sin(math.sqrt(10 * abs(x[i]))) / math.sqrt(10 * abs(x[i])))
            else:
                y.append(0)
        return y

    if task_num == 16:
        y = []
        for i in range(0, len(x)):
            y.append(2 * x[i] + 20 * math.pi * math.sin(2 * math.pi * x[i]))
        return y

    if task_num == 17:
        y = []
        for i in range(0, len(x)):
            y.append(x[i] / 2000)
            z = 1
            for j in range(0, len(x)):
                if j != i:
                    z *= math.cos(x[j])
                else:
                    z *= -1 * math.sin(x[i])
            y[i] += z
        return y