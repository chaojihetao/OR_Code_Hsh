"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : test3.py
@File : LRP.py

"""
from gurobipy import *
import numpy as np
from matplotlib import pyplot as plt


# 输入数据类
class Data:
    customNum = 0  # 客户数量
    nodeNum = 0  # 节点数
    vehicleNum = 0  # 车辆数
    depotNum = 0  # 设施数
    maxTime = 0  # 最大出行时间
    capacity = 0  # 车辆容量
    d_k = {}  # 车库对应关系
    cor_x = []  # 横坐标
    cor_y = []  # 纵坐标
    demand = []  # 需求
    service = []  # 服务时间
    startTime = []  # 开始时间
    endTime = []  # 结束时间
    timeDisMatrix = [[]]  # 时间矩阵
    disMatrix = [[]]  # 距离矩阵
    f_cost = []  # 选址成本


# 输出数据类
class Solution:
    objVal = 0  # 目标值
    routes = {}  # 路径


def readData(data, path):
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    count = 0
    for line in lines:
        count = count + 1
        line = line.strip()  # 去除前后空格
        strs = line.split()  # 切片
        if count == 1:
            data.customNum = int(strs[2])
            data.depotNum = int(strs[3])
            data.vehicleNum = int(strs[1]) * data.depotNum
            data.nodeNum = data.customNum + 2 * data.depotNum  # 节点复制
        if count == 2:
            data.maxTime = int(strs[0])
            data.capacity = int(strs[1])
        if count >= 6:
            data.cor_x.append(float(strs[1]))
            data.cor_y.append(float(strs[2]))
            data.service.append(float(strs[3]))
            data.demand.append(float(strs[4]))
            data.startTime.append(float(strs[-2]))
            data.endTime.append(float(strs[-1]))

    # 选址成本
    data.f_cost = [2000, 4000, 6000, 8000]

    # 复制节点
    for l in range(data.depotNum):
        data.cor_x.insert(l, data.cor_x[l - data.depotNum])
        data.cor_y.insert(l, data.cor_y[l - data.depotNum])
        data.service.insert(l, data.service[l - data.depotNum])
        data.demand.insert(l, data.demand[l - data.depotNum])
        data.startTime.insert(l, data.startTime[l - data.depotNum])
        data.endTime.insert(l, data.endTime[l - data.depotNum])

    # 车库对应关系
    itea = 0
    for l in range(data.depotNum):
        num = int(data.vehicleNum / data.depotNum)
        data.d_k[l] = [k for k in range(itea, itea + num)]
        itea += num

    # 距离矩阵
    data.disMatrix = np.zeros((data.nodeNum, data.nodeNum))
    data.timeDisMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            data.disMatrix[i, j] = math.sqrt(
                (data.cor_x[i] - data.cor_x[j]) ** 2 + (data.cor_y[i] - data.cor_y[j]) ** 2)
            data.timeDisMatrix[i, j] = data.disMatrix[i, j]

    return data


def solve(data, solution):
    # 建立模型
    model = Model('MDVRPTW')
    big_M = 9999

    # 设置变量
    x = {}
    s = {}
    y = {}
    w = {}
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            s[i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='s' + str(i) + str(k))
            for j in range(data.nodeNum):
                if i != j:
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name='x' + str(i) + str(j) + str(k))
    for l in range(data.depotNum):
        w[l] = model.addVar(vtype=GRB.BINARY, name='w' + str(l))
        for i in range(data.nodeNum):
            y[l, i] = model.addVar(vtype=GRB.BINARY, name='y' + str(l) + str(i))
    model.update()

    # 目标函数
    obj1 = quicksum(data.f_cost[l] * w[l] for l in range(data.depotNum))
    obj2 = quicksum(data.disMatrix[i, j] * x[i, j, k]
                    for i in range(data.nodeNum) for j in range(data.nodeNum) for k in range(data.vehicleNum) if i != j)
    model.setObjective(obj1 + obj2, GRB.MINIMIZE)

    # 约束条件
    # 1）车库相关约束
    for l in range(data.depotNum):
        demand_list = list(range(data.depotNum, data.nodeNum - data.depotNum))
        out_list = demand_list + [data.nodeNum - data.depotNum + l]
        in_list = demand_list + [l]
        # 变量约束
        for i in demand_list:
            model.addConstr(quicksum(x[i, j, k] for j in out_list for k in data.d_k[l] if j != i) == y[l, i],
                            name='variable' + str(l) + str(i))

        for k in data.d_k[l]:
            # 起点与终点的出入度
            model.addConstr(quicksum(x[l, j, k] for j in range(data.depotNum, data.nodeNum - data.depotNum)) == 1,
                            name='const_oj' + str(l))
            model.addConstr(quicksum(x[j, data.nodeNum - data.depotNum + l, k]
                                     for j in range(data.depotNum, data.nodeNum - data.depotNum)) == 1,
                            name='const_jo')
            # 中间点出入度平衡
            for h in demand_list:
                model.addConstr(quicksum(x[i, h, k] for i in in_list if i != h)
                                == quicksum(x[h, i, k] for i in out_list if i != h),
                                name='balance' + str(h))

    for k in range(data.vehicleNum):
        # ---容量约束---
        model.addConstr(
            quicksum(x[i, j, k] * data.demand[i] for i in range(data.nodeNum) for j in range(data.nodeNum) if i != j)
            <= data.capacity,
            name='capacity' + str(k))
        # ---服务时间约束---
        model.addConstr(
            quicksum(x[i, j, k] * data.service[i] for i in range(data.nodeNum) for j in range(data.nodeNum) if i != j)
            <= data.maxTime,
            name='maxTime' + str(k))

    # 3）需求点相关约束
    # 每个需求点都需要被访问
    for i in range(data.depotNum, data.nodeNum - data.depotNum):
        model.addConstr(quicksum(y[l, i] for l in range(data.depotNum)) == 1,
                        name='demand' + str(i))

    # model.addConstr(quicksum(x[i, j, k] for j in range(data.nodeNum) for k in range(data.vehicleNum) if i != j) == 1)

    # 4）破圈约束--时间约束
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            for j in range(data.nodeNum):
                if i != j:
                    model.addConstr(s[i, k] + data.service[i] + data.timeDisMatrix[i, j]
                                    <= s[j, k] + (1 - x[i, j, k]) * big_M,
                                    name='subTour' + str(k) + str(i) + str(j))

    # 5）硬时间窗约束
    for k in range(data.vehicleNum):
        for i in range(data.nodeNum):
            model.addConstr(data.startTime[i] <= s[i, k])
            model.addConstr(s[i, k] <= data.endTime[i])

    # 6) 选址约束
    for l in range(data.depotNum):
        for i in range(data.nodeNum):
            model.addConstr(y[l, i] <= w[l])

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        # ----Solution----
        # 目标值
        solution.objVal = model.objval

        # 路径
        for l in range(data.depotNum):
            for k in data.d_k[l]:
                i = l
                subRoute = [i]
                while i != data.nodeNum - data.depotNum + l:
                    for j in range(data.nodeNum):
                        if i != j:
                            if x[i, j, k].x > 0.5:
                                if j == data.nodeNum - data.depotNum + l:
                                    subRoute.append(l)
                                else:
                                    subRoute.append(j)
                                i = j
                solution.routes[l, k] = subRoute
    else:
        model.computeIIS()
        model.write('model.ilp')
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)

    return solution


def printSolution(solution):
    print('下面开始打印求解结果\n')
    print('ObjVal = %6.2f' % solution.objVal)

    print('\n---Route---')
    for l in range(data.depotNum):
        for k in data.d_k[l]:
            print('SubRoute', (l + 1, k + 1), ':', solution.routes[l, k])


def drawGraph(data, solution):
    fig = plt.figure(0)  # 创建空图
    plt.xlabel('cor_x')
    plt.ylabel('cor_y')
    pltName = 'MDVRPTW' + '_' + 'V' + str(data.vehicleNum) + '_' + 'C' + str(data.customNum)
    plt.title(pltName)

    # 绘制点集
    plt.scatter(data.cor_x[0:data.depotNum], data.cor_y[0:data.depotNum], color='red', alpha=1, marker=',',
                linewidths=2, label='depot')
    plt.scatter(data.cor_x[data.depotNum:data.nodeNum - data.depotNum],
                data.cor_y[data.depotNum:data.nodeNum - data.depotNum], color='black', alpha=1, marker='o',
                linewidths=1, label='customer')

    # 绘制路径
    for l in range(data.depotNum):
        for k in data.d_k[l]:
            lenSubRoute = len(solution.routes[l, k]) - 1  # 最后的点不作为起点
            for i in range(lenSubRoute):
                a = solution.routes[l, k][i]
                b = solution.routes[l, k][i + 1]
                x = [data.cor_x[a], data.cor_x[b]]
                y = [data.cor_y[a], data.cor_y[b]]
                plt.plot(x, y, color='black', linewidth=1.5)

    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    path = 'pr01'
    data = Data()
    data = readData(data, path)  # 读取参数

    solution = Solution()
    solution = solve(data, solution)  # 求解
    printSolution(solution)
    drawGraph(data, solution)  # 绘制路径图
