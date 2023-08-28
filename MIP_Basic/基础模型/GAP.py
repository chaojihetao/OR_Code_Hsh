"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : Gurobi课程
@File : AP.py
@Author : Hsh
@Time : 2023-01-14 11:28

"""
from gurobipy import *

# 集合
I = 5
i_list = list(range(I))

# 参数
c = [[4, 8, 7, 15, 12],
     [7, 9, 17, 14, 10],
     [6, 9, 12, 8, 7],
     [6, 7, 14, 6, 10],
     [6, 9, 12, 10, 6]]

# 模型
model = Model('指派问题')

# 变量
x = {}
for i in i_list:
    for j in i_list:
        x[i, j] = model.addVar(vtype=GRB.BINARY,
                               name='x'+str(i)+str(j))

# 目标函数
z = quicksum(c[i][j]*x[i, j] for i in i_list for j in i_list)
model.setObjective(z, GRB.MINIMIZE)

# 约束条件
for j in i_list:
    model.addConstr(quicksum(x[i, j] for i in i_list) == 1, name='const1'+str(j))

for i in i_list:
    model.addConstr(quicksum(x[i, j] for j in i_list) == 1, name='const2'+str(i))

# 求解
model.update()
model.optimize()

# 结果输出
for var in model.getVars():
    if var.x > 0.5:
        print(var.varname)


