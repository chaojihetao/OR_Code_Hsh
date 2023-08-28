"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : TSP问题
@File : test.py
@Author : Hsh
@Time : 2023-07-31 17:18

"""
import pandas as pd

disMatrix = pd.read_excel('distance.xlsx', index_col=0, header=0)    # 列索引为0列，表头为0行
Node = disMatrix.columns.values.tolist()
disMatrix = disMatrix.to_numpy()  # 转为数组格式

# 输出矩阵
print()
for i in Node:
    for j in Node:
        print('{:.2f}'.format(disMatrix[i, j]), end='\t')
    print()

from gurobipy import *

# 模型
model = Model('TSP')
big_M = 11

# 变量
x = {}
for i in Node:
    for j in Node:
        if i != j:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name='x'+'_'+str(i)+'_'+str(j))
u = {}
for i in Node[1:]:
    u[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='u'+str(i))
model.update()

# 目标函数
obj = quicksum(disMatrix[i, j] * x[i, j] for i in Node for j in Node if i != j)
model.setObjective(obj, GRB.MINIMIZE)

# 约束条件
for i in Node:
    # 出度约束
    model.addConstr(quicksum(x[i, j] for j in Node if i != j) == 1, name='out' + str(i))
    # 入度约束
    model.addConstr(quicksum(x[j, i] for j in Node if i != j) == 1, name='in' + str(i))

for i in Node[1:]:
    for j in Node[1:]:
        if i != j:
            model.addConstr(u[i] + 1 - big_M * (1 - x[i, j]) <= u[j], name='MTZ'+str(i)+str(j))

model.update()
# model.write('model.lp')
model.optimize()

for var in model.getVars():
    if var.x > 0.5:
        print(var.varname)
