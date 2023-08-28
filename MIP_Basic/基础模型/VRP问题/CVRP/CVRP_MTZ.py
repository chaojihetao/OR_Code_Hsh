"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : test3.py
@File : CVRP_MTZ.py
@Author : Hsh
@Time : 2023-01-20 7:20

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
    vehicleNum = 0
    capacity = 0
    cor_x = []
    cor_y = []
    demand = []
    disMatirx = [[]]


# 输出数据类
class Solution:
    Objval = 0
    X = []
    routes = {}


def readData(path, data, customNum, vehicleNum):
    data.vehicleNum = vehicleNum
    data.customNum = customNum
    data.nodeNum = customNum + 2  # 复制一份起点

    f = open(path, 'r')  # 只读模式打开数据路径文件
    lines = f.readlines()  # 以行的方式进行读取
    count = 0
    for line in lines:
        count = count + 1
        if count == 5:
            line = line[:-1]  # 去除空格
            str = re.split(r" +", line)  # 去除间隔
            data.capacity = float(str[2])
        elif 10 <= count <= 10 + customNum:
            line = line[:-1]  # 去除空格
            str = re.split(r" +", line)  # 去除间隔
            data.cor_x.append(float(str[2]))
            data.cor_y.append(float(str[3]))
            data.demand.append(float(str[4]))

    # 复制一份起点数据
    data.cor_x.append(data.cor_x[0])
    data.cor_y.append(data.cor_y[0])
    data.demand.append(data.demand[0])

    # 计算距离矩阵
    data.disMatirx = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            data.disMatirx[i, j] = math.sqrt(
                (data.cor_x[i] - data.cor_x[j]) ** 2 + (data.cor_y[i] - data.cor_y[j]) ** 2)

    return data


def printData(data):
    # 打印数据——注意起点是否被复制
    print("下面打印数据\n")
    print("vehicle number = %4d" % data.vehicleNum)
    print("vehicle capacity = %4d" % data.capacity)
    for i in range(len(data.demand)):
        print(data.demand[i])

    print("-------距离矩阵-------\n")
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            print("%6.2f" % (data.disMatirx[i][j]), end=" ")
        print()


def printSolution(solution):
    print('下面开始打印求解结果\n')
    print('ObjVal = %6.2f' % solution.Objval)

    print('\n---Route---')
    for k in range(data.vehicleNum):
        print('SubRoute', k + 1, ':', solution.routes[k])


def drawGraph(data, solution):
    fig = plt.figure(0)  # 创建空图
    plt.xlabel('cor_x')
    plt.ylabel('cor_y')
    pltName = 'C101' + '_' + str(data.vehicleNum) + '_' + str(data.customNum)
    plt.title(pltName)

    # 绘制点集
    plt.scatter(data.cor_x[0], data.cor_y[0], color='red', alpha=1, marker=',', linewidths=4, label='depot')
    plt.scatter(data.cor_x[1:-1], data.cor_y[1:-1], color='black', alpha=1, marker='o', linewidths=3, label='customer')

    # 绘制路径
    for k in range(data.vehicleNum):
        lenSubRoute = len(solution.routes[k]) - 1  # 最后的点不作为起点
        for i in range(lenSubRoute):
            a = solution.routes[k][i]
            b = solution.routes[k][i + 1]
            x = [data.cor_x[a], data.cor_x[b]]
            y = [data.cor_y[a], data.cor_y[b]]
            plt.plot(x, y, color='black', linewidth=1)

    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


def solver(data, solution):
    # 建立模型
    model = Model('CVRP_MTZ')

    # 设置变量
    x = {}
    u = {}
    for i in range(data.nodeNum):
        u[i] = model.addVar(lb=0,
                            vtype=GRB.CONTINUOUS,
                            name='u' + str(i))
        for j in range(data.nodeNum):
            for k in range(data.vehicleNum):
                x[i, j, k] = model.addVar(vtype=GRB.BINARY,
                                          name='x' + str(i) + str(j) + str(k))

    # 目标函数
    obj = quicksum(data.disMatirx[i][j] * x[i, j, k]
                   for i in range(data.nodeNum) for j in range(data.nodeNum) for k in range(data.vehicleNum))
    model.setObjective(obj, GRB.MINIMIZE)

    # 约束条件
    # 1）车辆相关约束
    for k in range(data.vehicleNum):
        # ---流平衡约束---
        # 起点与终点的出入度
        model.addConstr(quicksum(x[0, j, k] for j in range(1, data.nodeNum - 1)) == 1, name='const_0j')
        model.addConstr(quicksum(x[j, data.nodeNum - 1, k] for j in range(1, data.nodeNum - 1)) == 1, name='const_j0')
        # 中间点出入度平衡
        for h in range(1, data.nodeNum - 1):
            model.addConstr(quicksum(x[i, h, k] for i in range(data.nodeNum - 1))
                            == quicksum(x[h, i, k] for i in range(1, data.nodeNum)))
        # 防止原地转圈
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i == j:
                    model.addConstr(x[i, j, k] == 0)

        # ---容量约束---
        model.addConstr(quicksum(x[i, j, k] * data.demand[i] for i in range(data.nodeNum) for j in range(data.nodeNum))
                        <= data.capacity)

    # 2）需求点相关约束
    for i in range(1, data.nodeNum - 1):
        # 每个需求点都需要被访问
        model.addConstr(quicksum(x[i, j, k] for j in range(data.nodeNum) for k in range(data.vehicleNum)) == 1)

    # 3）破圈约束
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                model.addConstr(u[i] - u[j] + data.nodeNum * x[i, j, k] <= data.nodeNum - 1)

    model.update()
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        # ----Solution----
        # 目标值
        solution.Objval = model.objval
        # 变量
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                for k in range(data.vehicleNum):
                    if x[i, j, k].x > 0.5:
                        solution.X.append(x[i, j, k].varname)
        # 路径
        for k in range(data.vehicleNum):
            i = 0
            subRoute = [i]
            while i != data.nodeNum - 1:
                for j in range(data.nodeNum):
                    if x[i, j, k].x > 0.5:
                        if j == data.nodeNum - 1:
                            subRoute.append(0)
                        else:
                            subRoute.append(j)
                        i = j
            solution.routes[k] = subRoute
    else:
        model.computeIIS()
        model.write('model.ilp')
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)

    return model


if __name__ == '__main__':
    # 超参数设置
    customNum = 18  # 需求点数
    vehicleNum = 3  # 车辆数

    # 读取数据
    data = Data()  # 创建空输入数据类
    path = 'c101.txt'  # 数据路径
    readData(path, data, customNum, vehicleNum)  # 读取数据
    printData(data)  # 打印数据

    # 问题求解
    solution = Solution()  # 创建空输出数据类
    solver(data, solution)  # 模型求解
    printSolution(solution)  # 结果输出
    drawGraph(data, solution)  # 绘制路径图
