"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : xz.py
@File : MTZ_new.py
@Author : Hsh
@Time : 2023-02-07 9:48

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
    itea_list = []
    UB_list = []
    LB_list = []


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
    master_P.optimize()

    itea = 0  # 迭代次数
    UB = 25000  # 上界值--整体问题的可行解
    LB = 0  # 下界值--松弛问题的最优解

    while 1:
        itea += 1
        solution.itea_list.append(itea)
        solution.UB_list.append(UB)
        solution.LB_list.append(LB)
        print('\n!!!')
        print('Itea：', itea)
        print('fea:', UB)
        print('LB:', LB)
        print('!!!\n')

        apply_y = master_P.getAttr('X', y)  # 获取主问题变量

        """
        ----子问题模型----
        
        注意事项：
        1）直接采用子问题进行cut构造时：子问题最优解添加最优割；子问题无解(GRB.Status.INFEASIBLE)，对偶问题无界，添加可行割；
        另外，子问题必须严格按照标准形式进行书写，最小化问题--大于等于；最大化问题--小于等于。(可行割，获取const.FarkasDual)
        2）利用子问题对偶问题进行cut构造时，注意判断条件为，对偶问题无界（GRB.Status.UNBOUNDED),获取(const.UnbdRay)
        """
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
                constraint2[i, j] = sub_P.addConstr(-x[i, j] >= -apply_y[i], name='constraint2' + str(i) + str(j))

        # 求解子问题，添加约束
        sub_P.optimize()
        if sub_P.status == GRB.Status.OPTIMAL:
            opt_a = sub_P.getAttr('Pi', constraint1)
            opt_b = sub_P.getAttr('Pi', constraint2)
            master_P.addConstr(quicksum(opt_a[j] for j in range(data.J))
                               - quicksum(opt_b[i, j] * y[i] for i in range(data.I) for j in range(data.J)) <= z)
            # 更新上界
            Qy = sum(data.fCost[i] * apply_y[i] for i in range(data.I)) + sub_P.objval
            UB = min(Qy, UB)
        elif sub_P.status == GRB.Status.INFEASIBLE:
            und_a = sub_P.getAttr('FarkasDual', constraint1)
            und_b = sub_P.getAttr('FarkasDual', constraint2)
            master_P.addConstr(quicksum(und_a[j] for j in range(data.J))
                               - quicksum(und_b[i, j] * y[i] for i in range(data.I) for j in range(data.J)) >= 0)
        else:
            print(sub_P.status)

        master_P.update()
        master_P.optimize()
        LB = max(LB, master_P.objval)

        # 退出条件
        epsion = 0.01
        if UB - LB < epsion:
            break

    solution.objVal = master_P.objval
    return solution


def drawGraph(solution):
    fig = plt.figure(0)  # 创建空图
    plt.xlabel('Itea_Number')
    plt.ylabel('ObjValue')
    pltName = 'Benders Decomposition'
    plt.title(pltName)

    # 绘制曲线
    plt.plot(solution.itea_list, solution.LB_list, color='green', label='LB')
    plt.plot(solution.itea_list, solution.UB_list, color='red', label='UB')

    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    data = Data()
    path = 'B1.txt'
    data = readData(data, path)

    solution = Solution()
    solution = Benders(data, solution)  # 求解过程
    drawGraph(solution)  # 绘制收敛曲线
