"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : Classicial_Benders.py
@File : callback_Benders.py
@Author : Hsh
@Time : 2023-02-09 15:29

"""
from gurobipy import *
import numpy as np
from matplotlib import pyplot as plt


# 输入数据类
class Data:
    I = 0
    J = 0
    fCost = []
    cost = []


# 输出数据类
class Solution:
    objVal = 0


def readData(data, path):
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    count = 0
    for line in lines:
        count += 1
        if count == 2:
            line = line[:-1].split()
            data.I = int(line[0])
            data.J = int(line[1])
        elif count >= 3:
            line = line[:-1].split()
            data.fCost.append(float(line[1]))
            temp = []
            for j in range(data.J):
                temp.append(float(line[j + 2]))
            data.cost.append(temp)

    data.cost = np.array(data.cost)

    return data


def lazyConstraint(model, where):
    if where == GRB.Callback.MIPSOL:  # 当主问题获得整数可行解时，进行回调
        sol = model.cbGetSolution(model._y)  # 获取主问题的可行解,并作为参数输入子问题中

        # 构造子问题
        sub_P = Model('SP')
        sub_P.Params.OutputFlag = 0  # 不进行日志输出
        sub_P.Params.InfUnbdInfo = 1  # 获取极射线

        # 变量
        x = {}
        for i in range(data.I):
            for j in range(data.J):
                x[i, j] = sub_P.addVar(vtype=GRB.CONTINUOUS, name='x' + str(i) + str(j))
        sub_P.update()
        # 目标函数
        sub_z = quicksum(data.cost[i, j] * x[i, j] for i in range(data.I) for j in range(data.J))
        sub_P.setObjective(sub_z, GRB.MINIMIZE)
        # 约束条件
        constraint1 = {}
        constraint2 = {}
        for j in range(data.J):
            constraint1[j] = sub_P.addConstr(quicksum(x[i, j] for i in range(data.I)) == 1, name='constraint1' + str(j))
            for i in range(data.I):
                constraint2[i, j] = sub_P.addConstr(-x[i, j] >= -sol[i], name='constraint2' + str(i) + str(j))

        # 求解子问题，添加约束
        sub_P.optimize()
        if sub_P.status == GRB.Status.OPTIMAL:
            opt_a = sub_P.getAttr('Pi', constraint1)
            opt_b = sub_P.getAttr('Pi', constraint2)
            model.cbLazy(quicksum(opt_a[j] for j in range(data.J))
                         - quicksum(opt_b[i, j] * model._y[i] for i in range(data.I) for j in range(data.J)) <= model._z)
        elif sub_P.status == GRB.Status.INFEASIBLE:
            und_a = sub_P.getAttr('FarkasDual', constraint1)
            und_b = sub_P.getAttr('FarkasDual', constraint2)
            model.cbLazy(quicksum(und_a[j] for j in range(data.J)) - quicksum(
                und_b[i, j] * model._y[i] for i in range(data.I) for j in range(data.J)) >= 0)
        else:
            print(sub_P.status)


def Benders(data, solution):
    print('\n-----下面开始Benders求解------')
    """
    ----主问题模型----
    """
    master_P = Model('MP')
    # master_P.Params.OutputFlag = 0  # 不进行日志输出
    z = master_P.addVar(lb=0, obj=1, vtype=GRB.CONTINUOUS, name='z')  # 子问题对应变量
    y = {}
    for i in range(data.I):
        y[i] = master_P.addVar(obj=data.fCost[i], vtype=GRB.BINARY, name='y' + str(i))  # 变量

    master_P.update()

    # 为callback准备相关参数
    master_P._y = y
    master_P._z = z
    master_P.Params.LazyConstraints = 1  # 开通lazy constraint 的获取权限
    master_P.optimize(lazyConstraint)

    solution.objVal = master_P.objval

    return solution


if __name__ == '__main__':
    data = Data()
    path = 'B1.txt'
    data = readData(data, path)

    solution = Solution()
    solution = Benders(data, solution)  # 求解过程
