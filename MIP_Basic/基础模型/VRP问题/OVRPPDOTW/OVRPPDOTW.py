"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : Milk.py
@File : OVRPPDOTW.py
@Author : Hsh
@Time : 2023-02-22 14:27

"""
from gurobipy import *
import numpy as np
from matplotlib import pyplot as plt
import math


class Data:
    nodeNum = 0
    vehicleNum = 0
    capacity = 0
    speed = 0
    overTimeCost = 0
    transCost = 0
    peoCost = 0
    x = []
    y = []
    demand = []
    serviceTime = []
    exceptTime = []
    endTime = []
    disMatrix = []
    TimeMatrix = []


class Result:
    objValue = 0
    varList = []


def setData(path, data):
    data.nodeNum = 10
    data.vehicleNum = 2
    data.capacity = 10
    data.speed = 0.25
    data.overTimeCost = 0.6
    data.transCost = 4
    data.peoCost = 10

    # 读取文档数据
    f = open(path, encoding='utf-8')
    lines = f.readlines()
    count = 0
    for line in lines:
        count += 1
        if count >= 2:
            out = line.strip().split(r',')  # 去除首尾，切片,此时所有数据为string格式
            data.x.append(float(out[1]))
            data.y.append(float(out[2]))
            data.demand.append(float(out[3]))
            data.serviceTime.append(float(out[4]))
            data.exceptTime.append(float(out[5]))
            data.endTime.append(float(out[6]))

    # 计算距离与时间矩阵
    data.disMatrix = np.zeros((data.nodeNum, data.nodeNum))
    data.TimeMatrix = data.disMatrix
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            data.disMatrix[i, j] = math.sqrt((data.x[i] - data.x[j]) ** 2
                                             + (data.y[i] - data.y[j]) ** 2)
            data.TimeMatrix[i, j] = data.disMatrix[i, j] / data.speed

    return data


def solve(data, result):
    model = Model()
    big_M = 99999
    D = [0, 1]  # 配送员位置索引
    S = [2, 3, 4, 5]  # 商家索引
    C = [6, 7, 8, 9]  # 客户索引
    a_ij = {(i, j): 0 for i in S for j in C}
    for h in range(4):
        a_ij[S[h], C[h]] = 1

    # 设置变量
    z = {}
    x = {}
    t = {}
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            t[i, k] = model.addVar(name='t'+str(i)+str(k))
            for j in range(data.nodeNum):
                if i != j:
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name='x'+str(i)+str(j)+str(k))
        for i in S + C:
            z[i, k] = model.addVar(vtype=GRB.BINARY, name='z'+str(i)+str(k))

    # 目标函数
    obj = quicksum(data.disMatrix[i, j] * x[i, j, k]
                   for i in range(data.nodeNum)
                   for j in range(data.nodeNum)
                   for k in range(data.vehicleNum)
                   if i != j)
    model.setObjective(obj)

    # 约束条件
    for k in range(data.vehicleNum):
        model.addConstr(quicksum(x[i, j, k] for i in D for j in S + C if i != j) == 1,
                        name='c1'+str(k))
        model.addConstr(quicksum(z[i, k] for i in C) == 1, name='c2' + str(k))
        model.addConstr(quicksum(z[i, k] for i in S) == 0, name='c2.1' + str(k))
        for i in S + C:
            model.addConstr(quicksum(x[i, j, k] for j in range(data.nodeNum) if i != j) + z[i, k]
                            == quicksum(x[j, i, k] for j in range(data.nodeNum) if j != i), name='c3'+str(k))
        for i in S:
            for j in C:
                model.addConstr(t[i, k] + data.serviceTime[i] + data.TimeMatrix[i, j]
                                <= t[j, k] + big_M * a_ij[i, j], name='c4'+str(k))
                model.addConstr(t[j, k] <= data.endTime[j])
                model.addConstr(quicksum(x[i, h, k] for h in range(data.nodeNum) if i != h)
                                <= quicksum(x[h, j, k] for h in range(data.nodeNum) if j != h)
                                + big_M * (1 - a_ij[i, j]))

        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i != j:
                    model.addConstr(t[i, k] + data.serviceTime[i] + data.TimeMatrix[i, j]
                                    <= t[j, k] + big_M * (1 - x[i, j, k]), name='c5'+str(k))

        model.addConstr(quicksum(data.demand[i] * x[i, j, k]
                                 for i in range(data.nodeNum)
                                 for j in range(data.nodeNum)
                                 if i != j) <= data.capacity, name='c6'+str(k))

    for i in C + S:
        model.addConstr(quicksum(x[j, i, k] for j in range(data.nodeNum)
                                 if i != j for k in range(data.vehicleNum)) == 1,
                        name='c7'+str(i))
    for i in D:
        model.addConstr(quicksum(x[i, j, k] for j in C + S if i != j for k in range(data.vehicleNum)) == 1,
                        name='c8'+str(i))

    model.update()
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print(model.objval)
        for k in range(data.vehicleNum):
            for i in range(data.nodeNum):
                for j in range(data.nodeNum):
                    if i != j:
                        if x[i, j, k].x > 0.5:
                            print(x[i, j, k])
    else:
        model.computeIIS()
        model.write('model.ilp')
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)


if __name__ == '__main__':
    path = 'data.txt'
    data = Data()
    data = setData(path, data)
    result = Result()
    solve(data, result)
