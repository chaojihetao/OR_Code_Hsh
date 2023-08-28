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
    opt_list = []
    UB_list = []
    LB_list = []


# 定义读取函数
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
    print('\n------下面开始Benders求解------')
    """
    ------主问题模型------
    """
    master_P = Model('MP')
    z = master_P.addVar(lb=0, obj=1, vtype=GRB.CONTINUOUS, name='z')
    y = {}
    for i in range(data.I):
        y[i] = master_P.addVar(obj=data.fCost[i], vtype=GRB.BINARY, name='y' + str(i))
    master_P.update()
    master_P.optimize()

    itea = 0
    UB = 25000
    LB = 0

    while 1:
        itea += 1
        solution.itea_list.append(itea)
        solution.UB_list.append(UB)
        Solution.LB_list.append(LB)
        print('\n!!!')
        print('Itea:', itea)
        print('fea', UB)
        print('LB', LB)
        print('!!!\n')

        apply_y = master_P.getAttr('X', y)

        # stage = 1
        # if stage == 1:
        #     apply_y = [1] * data.I
        #     stage += 1
        # else:
        #     apply_y = master_P.getAttr('X', y)

        """
        子问题
        """
        sub_P = Model('SP')
        sub_P.Params.OutputFlag = 0
        sub_P.Params.InfUnbdInfo = 1

        # 变量
        a = {}
        for j in range(data.J):
            a[j] = sub_P.addVar(vtype=GRB.CONTINUOUS, name='a_s' + str(j))
        b = {}
        for j in range(data.J):
            for i in range(data.I):
                b[i, j] = sub_P.addVar(lb=0, vtype=GRB.CONTINUOUS, name='b' + str(i) + str(j))
        sub_P.update()
        # 目标函数
        sub_z = quicksum(a[j] for j in range(data.J)) - quicksum(
            apply_y[i] * b[i, j] for i in range(data.I) for j in range(data.J))
        sub_P.setObjective(sub_z, GRB.MAXIMIZE)
        # 约束条件
        constraint1 = {}
        for j in range(data.J):
            for i in range(data.I):
                constraint1[i, j] = sub_P.addConstr(a[j] - b[i, j] <= data.cost[i, j],
                                                    name='constraint1' + str(i) + str(j))

        sub_P.optimize()
        apply_a = sub_P.getAttr('X', a)
        apply_b = sub_P.getAttr('X', b)
        if sub_P.status == GRB.Status.OPTIMAL:
            master_P.addConstr(quicksum(apply_a[j] for j in range(data.J)) - quicksum(
                y[i] * apply_b[i, j] for i in range(data.I) for j in range(data.J)) <= z)
            Qy = sum(data.fCost[i] * apply_y[i] for i in range(data.I)) + sub_P.objval
            UB = min(Qy, UB)
        elif sub_P.status == GRB.Status.UNBOUNDED:
            master_P.addConstr(quicksum(apply_a[j] for j in range(data.J)) - quicksum(
                y[i] * apply_b[i, j] for i in range(data.I) for j in range(data.J)) <= 0)
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
    fig = plt.figure(0)
    plt.xlabel('Itea_Number')
    plt.ylabel('ObjValue')
    pltName = 'Benders Decomposition'
    plt.title(pltName)

    plt.plot(solution.itea_list, solution.LB_list, color='green', label='LB')
    plt.plot(solution.itea_list, solution.UB_list, color='red', label='fea')

    plt.grid(False)
    plt.legend(loc='best')
    plt.show(block=True)


if __name__ == '__main__':
    data = Data()
    path = 'B1.txt'
    data = readData(data, path)

    solution = Solution()
    solution = Benders(data, solution)
    drawGraph(solution)
