"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Time : 2023-02-24 17:33

"""
from gurobipy import *
import numpy as np
import math
from matplotlib import pyplot as plt


# 输入数据类
class Data:
    # 参数输入
    vehicleNum = 0  # 车辆数
    capacity = 0  # 车辆容量
    customNum = 0  # 需求点数
    nodeNum = 0  # 节点数
    f = []  # 车辆选择成本
    cor_x = []
    cor_y = []
    demand = []
    disMatrix = []


# 输出数据类
class Solution:
    objVal = 0
    itea_list = []
    UB_list = []
    LB_list = []


# 数据设定函数
def setData(data):
    data.vehicleNum = 2
    data.capacity = 50
    data.customNum = 4
    data.nodeNum = data.customNum + 2
    data.f = [10, 20]

    # 节点数据
    node, data.cor_x, data.cor_y, data.demand = multidict({0: [10, 6, 0],
                                                           1: [33, 5, 17],
                                                           2: [45, 13, 19],
                                                           3: [50, 32, 14],
                                                           4: [39, 36, 13],
                                                           5: [10, 6, 0]})

    # 距离矩阵
    data.disMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            data.disMatrix[i, j] = math.sqrt(
                (data.cor_x[i] - data.cor_x[j]) ** 2 + (data.cor_y[i] - data.cor_y[j]) ** 2)

    return data


def Benders(data, solution):
    """
    ----获取子问题下界值----
    """
    sub = Model('SP')
    sub.Params.OutputFlag = 0  # 不进行日志输出
    big_M = 99999

    # 变量
    x = {}
    u = {}
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            u[i, k] = sub.addVar()  # 辅助变量
            for j in range(data.nodeNum):
                if i != j:
                    x[i, j, k] = sub.addVar(vtype=GRB.BINARY, name='x' + str(i) + str(j) + str(k))
        sub.update()

    # 目标函数
    sub_z = quicksum(data.disMatrix[i, j] * x[i, j, k]
                     for i in range(data.nodeNum) for j in range(data.nodeNum) if i != j for k in
                     range(data.vehicleNum))
    sub.setObjective(sub_z, GRB.MINIMIZE)

    # 约束条件
    for k in range(data.vehicleNum):
        for h in range(1, data.nodeNum - 1):
            sub.addConstr(quicksum(x[h, j, k] for j in range(1, data.nodeNum) if j != h)
                          == quicksum(x[j, h, k] for j in range(data.nodeNum - 1) if j != h))

        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i != j:
                    sub.addConstr(u[i, k] + data.demand[j] <= u[j, k] + big_M * (1 - x[i, j, k]))

    for i in range(1, data.nodeNum - 1):
        sub.addConstr(
            quicksum(x[j, i, k] for j in range(data.nodeNum - 1) for k in range(data.vehicleNum) if j != i) == 1)

    # 获取子问题的可行域最下界
    sub.update()
    sub.optimize()
    L = sub.objval
    print('子问题的可行域下界：', L)

    print('\n-----下面开始Benders求解------')
    """
    ----主问题模型----
    """
    master_P = Model('MP')
    z = master_P.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z')  # 子问题对应变量
    master_P.Params.OutputFlag = 0  # 不进行日志输出
    y = {}
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            y[i, k] = master_P.addVar(vtype=GRB.BINARY, name='y' + str(i) + str(k))  # 变量

    # 目标函数
    master_P.setObjective(
        quicksum(data.f[k] * y[i, k] for i in range(1, data.nodeNum - 1) for k in range(data.vehicleNum)) + z)

    # 约束条件
    for i in range(1, data.nodeNum - 1):
        master_P.addConstr(quicksum(y[i, k] for k in range(data.vehicleNum)) == 1,
                           name='master_const1_' + str(i))

    for k in range(data.vehicleNum):
        master_P.addConstr(quicksum(data.demand[i] * y[i, k] for i in range(1, data.nodeNum - 1)) <= data.capacity,
                           name='master_capacity_' + str(k))

    master_P.update()
    master_P.optimize()

    # 初始参数
    itea = 0  # 迭代次数
    UB = GRB.INFINITY  # 上界值--整体问题的可行解
    LB = -GRB.INFINITY  # 下界值--整体问题的松弛解

    while 1:
        # 获取主问题变量
        apply_z = z.x
        apply_y = master_P.getAttr('X', y)

        # 选取当前 0-1 布局
        j1_list = [(i, k) for i in range(1, data.nodeNum - 1) for k in range(data.vehicleNum) if apply_y[i, k] == 1]
        j0_list = [(i, k) for i in range(1, data.nodeNum - 1) for k in range(data.vehicleNum) if apply_y[i, k] == 0]

        """
        ----子问题模型----
        """
        sub_P = Model('SP')
        sub_P.Params.OutputFlag = 0  # 不进行日志输出

        # 变量
        x = {}
        u = {}
        for k in range(data.vehicleNum):
            for i in range(data.nodeNum):
                u[i, k] = sub_P.addVar()  # 辅助变量
                for j in range(data.nodeNum):
                    if i != j:
                        x[i, j, k] = sub_P.addVar(vtype=GRB.BINARY, name='x' + str(i) + str(j))
        sub_P.update()

        # 目标函数
        sub_z = quicksum(data.disMatrix[i, j] * x[i, j, k]
                         for i in range(data.nodeNum) for j in range(data.nodeNum) if i != j
                         for k in range(data.vehicleNum))
        sub_P.setObjective(sub_z, GRB.MINIMIZE)

        # 约束条件
        for k in range(data.vehicleNum):
            for h in range(1, data.nodeNum - 1):
                sub_P.addConstr(quicksum(x[h, j, k]
                                         for j in range(1, data.nodeNum) if j != h)
                                == quicksum(x[j, h, k]
                                            for j in range(data.nodeNum - 1) if j != h))

            for i in range(data.nodeNum):
                for j in range(data.nodeNum):
                    if i != j:
                        sub_P.addConstr(u[i, k] + data.demand[j] <= u[j, k] + data.capacity * (1 - x[i, j, k]))

            for i in range(1, data.nodeNum - 1):
                sub_P.addConstr(quicksum(x[j, i, k] for j in range(data.nodeNum - 1) if i != j) == apply_y[i, k])

            sub_P.addConstr(quicksum(x[0, j, k] for j in range(1, data.nodeNum - 1)) == 1)
            sub_P.addConstr(quicksum(x[j, data.nodeNum - 1, k] for j in range(1, data.nodeNum - 1)) == 1)

        # 求解子问题，添加约束
        sub_P.optimize()
        if sub_P.status == GRB.Status.OPTIMAL:
            # 保存最优解，用于更新上界
            sub_obj = sub_P.objval
            new_ub = sub_obj + sum(data.f[k] * apply_y[i, k]
                                   for i in range(1, data.nodeNum - 1) for k in range(data.vehicleNum))
            UB = min(new_ub, UB)
            # 添加最优割
            if sub_P.objval > apply_z:
                feasible = quicksum(y[key] for key in j0_list) + quicksum((1 - y[key]) for key in j1_list)
                master_P.addConstr(z >= sub_obj - (sub_obj - L) * feasible, name='optimize_cut')
        else:
            # 添加可行割
            master_P.addConstr(quicksum(y[key] for key in j0_list) + quicksum((1 - y[key]) for key in j1_list) >= 1,
                               name='feasible_cut')

        # 主问题求解
        master_P.update()
        master_P.write('master.lp')
        master_P.optimize()

        # 更新上下界
        LB = max(master_P.objval, LB)

        # 获取相关数值
        itea += 1
        solution.itea_list.append(itea)
        solution.UB_list.append(UB)
        solution.LB_list.append(LB)
        print('\n!!!')
        print('Itea：', itea)
        print('fea:', UB)
        print('LB:', LB)
        print('!!!\n')

        # 退出条件
        epsilon = 0.01
        if UB - LB < epsilon:
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
    plt.plot(solution.itea_list, solution.LB_list, color='green', label='LB', marker='*')
    plt.plot(solution.itea_list, solution.UB_list, color='red', label='fea', marker='*')

    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    data = Data()
    data = setData(data)
    solution = Solution()
    solution = Benders(data, solution)
    drawGraph(solution)  # 绘制收敛曲线
