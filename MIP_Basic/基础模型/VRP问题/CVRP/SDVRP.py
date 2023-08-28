# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:37:34 2023

@author: 86178
"""

import re
from gurobipy import GRB, quicksum, Model
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import copy


# 输入数据类
class Data:
    customNum = 0
    nodeNum = 0
    vehicleNum = 0
    capacity = 0
    cor_x = []
    cor_y = []
    demand = []
    service = []
    startTime = []
    endTime = []
    disMatrix = [[]]
    timeMatrix = [[]]
    demand_min = 0


# 输出数据类
class Solution:
    Objval = 0
    X = []
    routes = {}  # 字典是因为每辆车都有路径


def setData(path, data, customNum, vehicleNum):
    data.vehicleNum = vehicleNum
    data.customNum = customNum
    data.nodeNum = customNum + 2  # 复制一份起点

    f = open(path, 'r')
    lines = f.readlines()
    count = 0
    for line in lines:
        count = count + 1
        if count == 5:
            line = line[:-1]  # 去掉空格
            str = re.split(r' +', line)
            data.capacity = float(str[2])
        elif 10 <= count <= 10 + customNum:
            line = line[:-1]  # 去掉空格
            str = re.split(r' +', line)
            data.cor_x.append(float(str[2]))
            data.cor_y.append(float(str[3]))
            data.demand.append(float(str[4]))
            data.startTime.append(float(str[5]))
            data.endTime.append(float(str[6]))
            data.service.append(float(str[7]))

    # 复制一份起点数据
    data.cor_x.append(data.cor_x[0])
    data.cor_y.append(data.cor_y[0])
    data.demand.append(data.demand[0])
    data.startTime.append(data.startTime[0])
    data.endTime.append(data.endTime[0])
    data.service.append(data.service[0])

    # 最小运输能力
    min_list = copy.deepcopy(data.demand[1:-1])
    min_list.append(data.capacity)
    data.demand_min = min(min_list)

    # 计算距离矩阵
    data.disMatrix = np.zeros((data.nodeNum, data.nodeNum))
    data.timeMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            data.disMatrix[i, j] = math.sqrt((data.cor_x[i] - data.cor_x[j]) ** 2
                                             + (data.cor_y[i] - data.cor_y[j]) ** 2)
            data.timeMatrix[i, j] = data.disMatrix[i, j]
    return data


def printData(data):
    # 打印数据，注意起点是否被复制
    print('\n下面开始打印数据')
    print('vehicle number = %4d' % data.vehicleNum)
    print('vehicle capacity = %4d' % data.capacity)
    for i in range(len(data.demand)):
        print(data.demand[i])

    print('\n距离矩阵')
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            print('%0.2f' % data.disMatrix[i, j], end='\t')
        print()


def printSolution(solution):
    print('\n下面是旅行路径：')
    print('ObjVal = %6.2f' % solution.Objval)

    print('\n---Route---')
    for k in range(data.vehicleNum):
        print('SubRoute', k + 1, ':', solution.routes[k])


def drawGraph(data, solution):
    fig = plt.figure(0)
    plt.xlabel('x')
    plt.ylabel('y')
    pltName = 'C101' + '_' + str(data.vehicleNum) + '_' + str(data.customNum)
    plt.title(pltName)

    # 绘制点集
    plt.scatter(data.cor_x[0], data.cor_y[0], color='red', alpha=1, marker=',', linewidths=4, label='Depot')
    plt.scatter(data.cor_x[1:-1], data.cor_y[1:-1], color='black', alpha=1, marker='o', linewidths=3, label='Customer')

    # 绘制路径
    for k in range(data.vehicleNum):
        lenSubRoute = len(solution.routes[k]) - 1  # 对i来说没有最后一个点
        for i in range(lenSubRoute):
            a = solution.routes[k][i]
            b = solution.routes[k][i + 1]
            x = [data.cor_x[a], data.cor_x[b]]
            y = [data.cor_y[a], data.cor_y[b]]
            plt.plot(x, y, linewidth=1.5, color='black')

    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


def solve(data, solution):
    # 建立模型
    model = Model('SDVRP')
    big_M = 9999

    # 设置变量
    x = {}
    a = {}  # 车k运输的量
    s = {}
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            s[i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='s' + str(i) + str(k))
            a[i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='a' + str(i) + str(k))
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i != j:
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY,
                                              name='x' + str(i) + str(j) + str(k))
    # 目标函数
    obj = quicksum(data.disMatrix[i, j] * x[i, j, k]
                   for i in range(data.nodeNum)
                   for j in range(data.nodeNum)
                   if i != j
                   for k in range(data.vehicleNum))
    model.setObjective(obj, GRB.MINIMIZE)

    # 约束条件
    # 1> 车辆相关约束
    for k in range(vehicleNum):
        # ---流平衡约束---
        model.addConstr(quicksum(x[0, j, k] for j in range(1, data.nodeNum - 1)) == 1,
                        name='const_0j')
        model.addConstr(quicksum(x[j, data.nodeNum - 1, k] for j in range(1, data.nodeNum - 1)) == 1,
                        name='const_j0')
        # 中间点的出入度
        for h in range(1, data.nodeNum - 1):
            model.addConstr(quicksum(x[i, h, k] for i in range(data.nodeNum - 1) if i != h) ==
                            quicksum(x[h, i, k] for i in range(1, data.nodeNum) if i != h))

            # ---容量---
        model.addConstr(quicksum(a[i, k] for i in range(1, data.nodeNum - 1)) <= data.capacity)
        for i in range(1, data.nodeNum - 1):
            model.addConstr(quicksum(x[i, j, k] for j in range(1, data.nodeNum) if i != j) * data.demand_min >= a[i, k],
                            name='con14'+'_'+str(i))

    # 2>需求点相关约束
    for i in range(1, data.nodeNum - 1):
        # 每个需求点需要被访问
        # model.addConstr(quicksum(x[i, j, k] for j in range(data.nodeNum) for k in range(data.vehicleNum)) <= 1)
        model.addConstr(quicksum(a[i, k] for k in range(data.vehicleNum)) >= data.demand[i], name='con2'+'_'+str(i))
    # 3>破圈约束
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i != j:
                    model.addConstr(s[i, k] + data.service[i] + data.timeMatrix[i, j]
                                    <= s[j, k] + (1 - x[i, j, k]) * big_M,
                                    name='subtour' + str(k) + str(i) + str(j))
    # 4>硬时间窗约束
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            model.addConstr(data.startTime[i] <= s[i, k])
            model.addConstr(s[i, k] <= data.endTime[i])

    model.update()
    model.optimize()

    # 输出路径
    if model.status == GRB.Status.OPTIMAL:
        # ---solution---
        # 目标值
        solution.Objval = model.Objval
        # 变量
        for k in range(data.vehicleNum):
            for i in range(data.nodeNum):
                for j in range(data.nodeNum):
                    if i != j:
                        if x[i, j, k].x > 0.5:
                            solution.X.append(x[i, j, k].varname)
        print(solution.X)
        # 路径
        for k in range(data.vehicleNum):
            i = 0
            subRoute = [i]
            while i != data.nodeNum - 1:
                for j in range(data.nodeNum):
                    if j != i:
                        if x[i, j, k].x > 0.5:
                            if j == data.nodeNum - 1:
                                subRoute.append(0)
                            else:
                                subRoute.append(j)
                            i = j
            solution.routes[k] = subRoute
    else:
        model.computeIIS()
        model.write('SDVRP.ilp')
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)


# 主函数
if __name__ == '__main__':
    # 参数设置
    customNum = 14
    vehicleNum = 4

    # 读取数据
    path = 'c101.txt'
    data = Data()
    setData(path, data, customNum, vehicleNum)  # 函数,设置参数
    printData(data)  # 打印参数

    # 问题求解
    solution = Solution()
    solve(data, solution)  # 模型求解
    printSolution(solution)
    drawGraph(data, solution)   # 画图
