"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : test2.py
@File : TSP_MTZ.py
@Author : Hsh
@Time : 2023-01-21 12:35

"""
import re
from gurobipy import *
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# 输入数据类
class Data:
    customNum = 0
    nodeNum = 0
    cor_x = []
    cor_y = []
    disMatrix = [[]]


# 输出数据类
class Solution:
    route = []


def setData(path, data, customNum):
    data.customNum = customNum
    data.nodeNum = customNum + 2

    f = open(path, 'r')  # 只读模式打开文件
    lines = f.readlines()  # 按行进行切割数据
    count = 0
    for line in lines:
        count = count + 1
        if 10 <= count <= 10 + data.customNum:
            line = line[:-1]  # 去除尾随回车符
            str = re.split(r' +', line)  # 切割文档
            data.cor_x.append(float(str[2]))
            data.cor_y.append(float(str[3]))

    # 复制一份起点
    data.cor_x.append(data.cor_x[0])
    data.cor_y.append(data.cor_y[0])

    # 计算距离矩阵
    data.disMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            data.disMatrix[i, j] = math.sqrt(
                (data.cor_x[i] - data.cor_x[j]) ** 2 + (data.cor_y[i] - data.cor_y[j]) ** 2)

    return data


def printData(data):
    print('\n 下面开始打印数据')
    print('CustomNum:', data.customNum)
    print('\n距离矩阵')
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            print('%0.2f' % data.disMatrix[i, j], end='\t')
        print()


def solve(data, solution):
    big_M = 21
    model = Model('TSP_MTZ')

    # 设置变量
    u = {}
    x = {}
    for i in range(data.nodeNum):
        u[i] = model.addVar(lb=0,
                            vtype=GRB.INTEGER,
                            name='u' + str(i))
        for j in range(data.nodeNum):
            x[i, j] = model.addVar(vtype=GRB.BINARY,
                                   name='x' + str(i) + str(j))

    # 目标函数
    obj = quicksum(data.disMatrix[i, j] * x[i, j] for i in range(data.nodeNum) for j in range(data.nodeNum))
    model.setObjective(obj, GRB.MINIMIZE)

    # 约束条件
    for i in range(data.nodeNum - 1):
        model.addConstr(quicksum(x[i, j] for j in range(1, data.nodeNum)) == 1)

    for i in range(1, data.nodeNum):
        model.addConstr(quicksum(x[j, i] for j in range(data.nodeNum - 1)) == 1)

    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            model.addConstr(u[i] + 1 <= u[j] + (1 - x[i, j]) * big_M)

    model.addConstr(x[0, data.nodeNum - 1] == 0)  # 不能让起点直接到达终点

    model.update()
    model.optimize()

    # 获得路径
    i = 0
    solution.route = [i]
    while i != data.nodeNum - 1:
        for j in range(data.nodeNum):
            if x[i, j].x > 0.5:
                if j == data.nodeNum - 1:
                    solution.route.append(0)
                else:
                    solution.route.append(j)
                i = j
    print('Route:', solution.route)


def drawGraph(data, solution):
    fig = plt.figure(0)  # 创建空图
    plt.xlabel('cor_x')
    plt.ylabel('cor_y')
    pltName = 'TSP' + '_' + '_' + str(data.customNum)
    plt.title(pltName)

    # 绘制点集
    plt.scatter(data.cor_x[:-1], data.cor_y[:-1], color='blue', alpha=1, marker='o', linewidths=1, label='Node')

    # 绘制路径
    lenSubRoute = len(solution.route) - 1  # 最后的点不作为起点
    for i in range(lenSubRoute):
        a = solution.route[i]
        b = solution.route[i + 1]
        x = [data.cor_x[a], data.cor_x[b]]
        y = [data.cor_y[a], data.cor_y[b]]
        plt.plot(x, y, color='black', linewidth=1.5)

    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    customNum = 20  # 需求点数量

    # 导入数据
    path = 'c101.txt'
    data = Data()
    setData(path, data, customNum)  # 设置参数
    printData(data)  # 查看数据

    solution = Solution()  # 创建空数据类
    solve(data, solution)  # 模型求解

    drawGraph(data, solution)  # 绘制路由图
