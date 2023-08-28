"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : KP.py
@File : MTZ_new.py
@Author : Hsh
@Time : 2023-01-17 14:08

"""
from gurobipy import *
import numpy as np

# 参数
Node = ['S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'T']   # 点集合
A = {('S', 'A'): 1, ('S', 'B'): 1, ('A', 'C'): 5, ('A', 'D'): 4, ('B', 'D'): 7, ('B', 'E'): 3,
     ('C', 'F'): 2, ('C', 'G'): 1, ('D', 'G'): 1, ('D', 'H'): 2, ('E', 'H'): 5, ('E', 'I'): 4,
     ('F', 'K'): 3, ('G', 'K'): 4, ('G', 'L'): 4, ('H', 'L'): 2, ('H', 'M'): 4, ('I', 'M'): 2,
     ('K', 'N'): 5, ('L', 'N'): 2, ('L', 'O'): 8, ('M', 'O'): 4, ('N', 'T'): 2, ('O', 'T'): 1}
# 模型
model = Model('SPP')

# 变量
x = {}
for arc in A.keys():
    x[arc] = model.addVar(vtype=GRB.BINARY,
                          name='x' + str(arc))

# 目标
z = quicksum(A[arc] * x[arc] for arc in A.keys())
model.setObjective(z, GRB.MINIMIZE)

# 约束
model.addConstr(quicksum(x[arc] for arc in A.keys() if arc[0] == 'S')
                - quicksum(x[arc] for arc in A.keys() if arc[1] == 'S') == 1)

model.addConstr(quicksum(x[arc] for arc in A.keys() if arc[0] == 'T')
                - quicksum(x[arc] for arc in A.keys() if arc[1] == 'T') == -1)

for i in Node:
    if i != 'S' and i != 'T':
        model.addConstr(quicksum(x[arc] for arc in A.keys() if arc[0] == i)
                        == quicksum(x[arc] for arc in A.keys() if arc[1] == i))

# 求解
model.update()
model.optimize()
print(model.objval)

# 输出
start = 'S'     # 起点（前向节点）
path = [start]  # 最短路径列表
for key in A.keys():
    if x[key].x > 0.5:
        print(x[key].varname)
        start = key[1]  # 更新前向节点
        path.append(start)  # 将该点加入路径中
print('\n最短路径：{}'.format(path))     # 输出最短路径
