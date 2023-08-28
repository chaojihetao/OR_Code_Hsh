"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : Classicial_Benders.py
@File : coding.py
@Author : Hsh
@Time : 2023-05-13 17:44

"""

from gurobipy import *
import numpy as np
from matplotlib import pyplot as plt
import math


# 输入数据类
class Data:
    I = 50  # 设施数
    J = 100  # 需求点数
    cor_xI = []  # 设施横坐标
    cor_yI = []  # 设施纵坐标
    cor_xJ = []  # 需求点横坐标
    cor_yJ = []  # 需求点纵坐标
    fCost = []  # 选址成本
    cost = []  # 运输成本


# 输出数据类
class Solution:
    objVal = 0  # 目标值
    itea_list = []  # 迭代次数
    UB_list = []    # 上界值列表
    LB_list = []    # 下界值列表
    y_value = []    # 选择变量解
    x_value = np.zeros((50, 100))   # 运输变量解


# 读取数据函数
def readData(data, path_I, path_J):
    # 读取设施点数据（横坐标、纵坐标、设施选址成本）
    f = open(path_I, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        data.cor_xI.append(float(line[1]))  # 横坐标
        data.cor_yI.append(float(line[2]))  # 纵坐标
        data.fCost.append(float(line[3]))   # 选址成本
    f.close()

    # 读取需求点数据（横坐标、纵坐标）
    f = open(path_J, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        data.cor_xJ.append(float(line[1]))  # 横坐标
        data.cor_yJ.append(float(line[2]))  # 纵坐标
    f.close()

    # 计算距离矩阵
    data.cost = np.zeros((data.I, data.J))
    for i in range(data.I):
        for j in range(data.J):
            data.cost[i, j] = math.sqrt((data.cor_xI[i] - data.cor_xJ[j]) ** 2 + (data.cor_yI[i] - data.cor_yJ[j]) ** 2)

    return data


# 算法主体
def Benders(data, solution):
    print('\n-----下面开始Benders求解------')
    """
    ----主问题模型----
    """
    master_P = Model('MP')
    # master_P.Params.OutputFlag = 0  # 不进行日志输出
    z = master_P.addVar(lb=0, obj=1, vtype=GRB.CONTINUOUS, name='z')  # 子问题对应变量
    y = {}  # 选址变量——以字典的数据结构进行存储
    for i in range(data.I):
        y[i] = master_P.addVar(obj=data.fCost[i], vtype=GRB.BINARY, name='y' + str(i))  # 变量
    master_P.update()
    master_P.optimize()

    itea = 0  # 迭代次数
    UB = 2000  # 上界值--整体问题的可行解
    LB = 0  # 下界值--松弛问题的最优解

    while 1:
        # 每次迭代都进行输出（当前迭代次数、上界值、下界值）
        itea += 1
        solution.itea_list.append(itea)
        solution.UB_list.append(UB)
        solution.LB_list.append(LB)
        print('\n!!!')
        print('Itea：', itea)
        print('fea:', UB)
        print('LB:', LB)
        print('!!!\n')

        # 获取主问题变量
        apply_y = master_P.getAttr('X', y)

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
        # 添加最优割
        if sub_P.status == GRB.Status.OPTIMAL:
            opt_a = sub_P.getAttr('Pi', constraint1)
            opt_b = sub_P.getAttr('Pi', constraint2)
            master_P.addConstr(quicksum(opt_a[j] for j in range(data.J))
                               - quicksum(opt_b[i, j] * y[i] for i in range(data.I) for j in range(data.J)) <= z)
            # 更新上界
            Qy = sum(data.fCost[i] * apply_y[i] for i in range(data.I)) + sub_P.objval
            UB = min(Qy, UB)

        # 添加可行割
        elif sub_P.status == GRB.Status.INFEASIBLE:
            und_a = sub_P.getAttr('FarkasDual', constraint1)
            und_b = sub_P.getAttr('FarkasDual', constraint2)
            master_P.addConstr(quicksum(und_a[j] for j in range(data.J))
                               - quicksum(und_b[i, j] * y[i] for i in range(data.I) for j in range(data.J)) >= 0)

        # 如果无解，则运行冲突模型，输出冲突约束
        else:
            print(sub_P.status)
            sub_P.computeIIS()
            sub_P.write('model.ilp')
            for c in sub_P.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)

        master_P.update()
        master_P.optimize()
        LB = max(LB, master_P.objval)

        # 退出条件
        epsion = 1
        if UB - LB < epsion:
            break

    # 记录解的信息
    solution.objVal = master_P.objval
    for i in range(data.I):
        solution.y_value.append(y[i].x)
        for j in range(data.J):
            solution.x_value[i, j] = x[i, j].x

    # 输出解
    print('目标值：{}'.format(solution.objVal))
    print('选址结果：{}'.format(solution.y_value))
    print('分配结果：{}'.format(solution.x_value))

    return solution


# 收敛曲线绘制函数
def drawGraph(solution):
    fig = plt.figure(0)  # 创建空图
    plt.xlabel('Itea_Number')   # 横坐标轴
    plt.ylabel('ObjValue')      # 纵坐标轴
    pltName = 'Benders Decomposition'   # 图标题
    plt.title(pltName)

    # 绘制曲线
    plt.plot(solution.itea_list, solution.LB_list, color='green', label='LB')
    plt.plot(solution.itea_list, solution.UB_list, color='red', label='UB')

    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


# 运输路线图绘制函数
def drawPathGraph(data, solution):
    fig = plt.figure(0)  # 创建空图
    plt.xlabel('cor_x')
    plt.ylabel('cor_y')
    pltName = 'benders' + '_' + str(data.I) + '_' + str(data.J)
    plt.title(pltName)

    # 绘制点集
    plt.scatter(data.cor_xI, data.cor_yI, color='red', alpha=1, marker=',', linewidths=2, label='depot')
    plt.scatter(data.cor_xJ, data.cor_yJ, color='blue', alpha=1, marker='o', linewidths=1, label='customer')

    # 绘制路径
    for i in range(data.I):
        for j in range(data.J):
            x = [data.cor_xI[i], data.cor_xJ[j]]
            y = [data.cor_yI[i], data.cor_yJ[j]]
            plt.plot(x, y, color='black', linewidth=0.5, alpha=solution.x_value[i, j])

    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    data = Data()
    solution = Solution()
    path_I = 'I.txt'
    path_J = 'J.txt'

    data = readData(data, path_I, path_J)   # 输入数据
    solution = Benders(data, solution)  # 求解过程
    drawGraph(solution)  # 绘制收敛曲线
    drawPathGraph(data, solution)   # 绘制运输路线图
