"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : CVRP_MTZ.py
@File : test.py
@Author : Hsh
@Time : 2023-08-02 21:21

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
    disMatrix = [[]]


# 输出数据类
class Solution:
    ObjVal = 0
    X = []
    routes = {}


def readData(path, data, customNum, vehicleNum):
    data.vehicleNum = vehicleNum
    data.customNum = customNum
    data.nodeNum = customNum + 1  # 复制一份起点

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

    # # 复制一份起点数据
    # data.cor_x.append(data.cor_x[0])
    # data.cor_y.append(data.cor_y[0])
    # data.demand.append(data.demand[0])

    # 计算距离矩阵
    data.disMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            data.disMatrix[i, j] = math.sqrt(
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
            print("%6.2f" % (data.disMatrix[i][j]), end=" ")
        print()


def solver(data, solution):
    # 建立模型
    model = Model('CVRP_MTZ')

    # 设置变量
    x = {}
    u = {}
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i != j:
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY,
                                              name='x' + str(i) + str(j) + str(k))
        for i in range(1, data.nodeNum):
            u[i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                   name='u' + '_' + str(i) + '_' + str(k))
    model.update()

    # 目标函数
    obj = quicksum(data.disMatrix[i, j] * x[i, j, k]
                   for i in range(data.nodeNum) for j in range(data.nodeNum) for k in range(data.vehicleNum)
                   if i != j)
    model.setObjective(obj, GRB.MINIMIZE)

    # 约束条件
    # 1）车辆相关约束
    for k in range(data.vehicleNum):
        # ---流平衡约束---
        for i in range(data.nodeNum):
            model.addConstr(quicksum(x[i, j, k] for j in range(data.nodeNum) if i != j) == 1,
                            name='out' + str(i) + '_' + str(k))
            model.addConstr(quicksum(x[j, i, k] for j in range(data.nodeNum) if i != j) == 1,
                            name='in' + str(i) + '_' + str(k))

        # ---容量约束---
        model.addConstr(quicksum(x[i, j, k] * data.demand[i]
                                 for i in range(data.nodeNum) for j in range(data.nodeNum) if i != j)
                        <= data.capacity, name='capacity_constraint' + '_' + str(k))

    # 2）需求点相关约束
    for i in range(1, data.nodeNum):
        # 每个需求点都需要被访问
        model.addConstr(quicksum(x[j, i, k] for j in range(data.nodeNum) for k in range(data.vehicleNum) if i != j)
                        >= 1, name='demand_constraint' + '_' + str(i))

    # 3）破圈约束
    for k in range(data.vehicleNum):
        for i in range(1, data.nodeNum):
            for j in range(1, data.nodeNum):
                if i != j:
                    model.addConstr(u[i, k] - u[j, k] + data.nodeNum * x[i, j, k] <= data.nodeNum - 1)

    model.update()
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        # ----Solution----
        # 目标值
        solution.ObjVal = model.objval
        # 变量
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i != j:
                    for k in range(data.vehicleNum):
                        if x[i, j, k].x > 0.5:
                            solution.X.append(x[i, j, k].varname)
        # # 路径
        # for k in range(data.vehicleNum):
        #     i = 0
        #     subRoute = [i]
        #     while i != data.nodeNum - 1:
        #         for j in range(data.nodeNum):
        #             if x[i, j, k].x > 0.5:
        #                 if j == data.nodeNum - 1:
        #                     subRoute.append(0)
        #                 else:
        #                     subRoute.append(j)
        #                 i = j
        #     solution.routes[k] = subRoute
    else:
        model.computeIIS()
        model.write('model.ilp')
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)

    return model


if __name__ == '__main__':
    # 超参数设置
    customNum = 12  # 需求点数
    vehicleNum = 5  # 车辆数

    # 导入数据
    data = Data()  # 创建空输入数据类
    path = 'c101.txt'  # 数据路径
    readData(path, data, customNum, vehicleNum)  # 读取数据
    printData(data)  # 打印数据

    # 问题求解
    solution = Solution()  # 创建空输出数据类
    solver(data, solution)  # 模型求解

    # 输出结果
    # 绘制路径
