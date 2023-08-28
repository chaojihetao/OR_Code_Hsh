"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : Gurobi课程
@File : KP.py
@Author : Hsh
@Time : 2023-01-14 10:15

"""
# 导入模块包
from gurobipy import *

# 确定参数
I = [0, 1, 2, 3, 4]     # 物品集合
P = [4, 2, 1, 10, 2]    # 价值属性
W = [12, 2, 1, 4, 1]    # 重量属性
C = 15  # 背包重量限制

# 建立模型
model = Model('KP')

# 添加变量
x = {}
for i in I:
    x[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x'+str(i))
model.update()

# 设置目标函数
z = quicksum(P[i] * x[i] for i in I)
model.setObjective(z, GRB.MAXIMIZE)

# 约束条件
model.addConstr(quicksum(W[i] * x[i] for i in I) <= C, name='capacity_constraint')

# 求解模型
model.update()
model.optimize()

# 输出结果
for i in I:
    if x[i].x > 0.5:
        print(x[i].varname)