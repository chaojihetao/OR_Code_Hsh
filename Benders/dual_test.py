"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : xz.py
@File : dual_test.py
@Author : Hsh
@Time : 2022-12-24 13:39

"""
"""
该代码是通过小算例进行对偶模型的检验
如果子问题与其对偶问题的优化结果一致
根据强弱对偶定理，可以证明对偶模型构造正确
"""

from gurobipy import *

# 相关数据（小算例）
I = 5
J = 6
c = [[11, 12, 14, 17, 20, 25],
     [15, 16, 17, 14, 11, 24],
     [12, 16, 19, 18, 10, 9],
     [22, 25, 21, 14, 16, 12],
     [10, 16, 18, 19, 21, 24]]
y = [1, 0, 1, 0, 1]

# SP
SP = Model('子问题')
x = SP.addVars(I, J, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='X')

obj = quicksum(c[i][j] * x[i, j] for i in range(I) for j in range(J))
SP.setObjective(obj, GRB.MINIMIZE)

for j in range(J):
    SP.addConstr(quicksum(x[i, j] for i in range(I)) == 1)

for i in range(I):
    for j in range(J):
        SP.addConstr(x[i, j] <= y[i])

SP.update()
SP.optimize()

# dual_sp
dsp = Model('对偶子问题')
a = dsp.addVars(J, vtype=GRB.CONTINUOUS, name='a_s')
b = dsp.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='b')

dual_obj = quicksum(a[j] for j in range(J)) - quicksum(b[i, j] * y[i] for i in range(I) for j in range(J))
dsp.setObjective(dual_obj, GRB.MAXIMIZE)

for i in range(I):
    for j in range(J):
        dsp.addConstr(a[j] - b[i, j] <= c[i][j])

dsp.update()
dsp.optimize()


if SP.objVal == dsp.objVal:
    print('对偶问题构造正确！！！')