"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : xz.py
@File : callback.py
@Author : Hsh
@Time : 2022-12-24 13:11

"""
from gurobipy import *


def my_lazy_callback(model, where):
    if where == GRB.Callback.MIPSOL:  # 当主问题获得整数可行解时，进行回调
        sol = model.cbGetSolution(model._y)  # 获取主问题的可行解,并作为参数输入子问题中

        # 构造对偶子问题
        dsp = Model('对偶子问题')
        dsp.Params.outputflag = 0       # 不输出对偶子问题的求解日志
        dsp.Params.InfUnbdInfo = 1      # 获取极射线

        # 设置变量
        a = dsp.addVars(n, vtype=GRB.CONTINUOUS, name='a_s')
        b = dsp.addVars(m, n, lb=0, vtype=GRB.CONTINUOUS, name='b')

        # 定义目标函数
        dual_obj = quicksum(a[j] for j in range(n)) - quicksum(b[i, j] * sol[i] for i in range(m) for j in range(n))
        dsp.setObjective(dual_obj, GRB.MAXIMIZE)

        # 定义约束条件
        for i in range(m):
            for j in range(n):
                dsp.addConstr(a[j] - b[i, j] <= ccost[i][j])

        # 求解
        dsp.update()
        dsp.optimize()

        # 根据求解情况进行最优割与可行割的添加
        if dsp.status == GRB.OPTIMAL:
            model.cbLazy(quicksum(a[j].X for j in range(n)) - quicksum(b[i, j].X * model._y[i] for i in range(m) for j in range(n)) <= model._z)
        elif dsp.status == GRB.UNBOUNDED:
            model.cbLazy(quicksum(a[j].UnbdRay for j in range(n)) - quicksum(b[i, j].UnbdRay * model._y[i] for i in range(m) for j in range(n)) <= 0)


# 读取数据
f = open('B1.txt', 'r')
filemes = f.readline()
nums = f.readline().split()
m = eval(nums[0])
n = eval(nums[1])
fcost = []
ccost = []
for line in f.readlines():
    ls = line.split()
    fcost.append(eval(ls[1]))
    cost = []
    for i in range(1, n + 1):
        cost.append(eval(ls[i + 1]))
    ccost.append(cost)
f.close()

# 建模与求解
model = Model("主问题")  # 主问题模型

# 添加变量
y = model.addVars(m, vtype=GRB.BINARY, name='y')  # 0-1变量,是否被选，1表示选择，0表示不选择
z = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='z')  # 子问题目标关联

# 设置目标函数
obj = quicksum(fcost[j] * y[j] for j in range(m)) + z
model.setObjective(obj, GRB.MINIMIZE)

model.update()

# 为callback准备相关参数
model._y = y
model._z = z
model.Params.LazyConstraints = 1        # 开通lazy constraint 的获取权限

model.optimize(my_lazy_callback)        # 调用callback进行求解

# 输出结果
print('\n求解时间为：', model.runtime, '秒')
print("\n最终结果如下：")
# 选择结果
solution_y = []
for i in range(m):
    if y[i].X >= 0.5:
        solution_y.append(i + 1)
print('选址结果为：', end='\t')
print(solution_y, end='\n')
# 目标值
print("Obj = {:.2f}".format(model.objVal))