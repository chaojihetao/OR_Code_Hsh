"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : Gurobi课程
@File : SCP.py
@Author : Hsh
@Time : 2023-01-14 11:05

"""
from gurobipy import *

# 集合
I = 7
i_list = list(range(I))

# 参数
c = [3.1, 2.3, 3.7, 2.6, 1.8, 3.3, 1.9]
a = [[1, 1, 1, 0, 0, 0, 0],
     [1, 1, 1, 1, 0, 0, 0],
     [1, 1, 1, 1, 0, 0, 0],
     [0, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 1, 1, 0, 1],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1]]

# 模型
model = Model('集合覆盖')

# 变量
x = {}
for j in i_list:
    x[j] = model.addVar(vtype=GRB.BINARY,
                        name='x'+str(j))

# 目标
z = quicksum(c[j]*x[j] for j in i_list)
model.setObjective(z, GRB.MINIMIZE)

# 约束
for i in i_list:
    model.addConstr(quicksum(a[i][j]*x[j] for j in i_list) >= 1, name='const'+str(i))

# 求解
model.update()
model.optimize()

# 输出
for var in model.getVars():
    if var.x > 0.5:
        print(var.varname)