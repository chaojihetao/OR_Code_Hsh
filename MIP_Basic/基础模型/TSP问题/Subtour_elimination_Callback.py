"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@File : Sub-tour_elimination_Callback.py
@Time : 2022-11-12 16:10

"""
# 导入相关的库
from gurobipy import *
import pandas as pd
import numpy as np
from itertools import combinations, product


#  读取数据
node_list = list(np.arange(10))
distance = np.array(pd.read_excel('distance.xlsx', sheet_name='Sheet2'))
print('\n\n__输出距离矩阵__')
print(distance)

# python product ：依次取出第一个列表与第二个列表中的元素，组成元组
dist = {(node1, node2): distance[node1, node2] for node1, node2 in product(node_list, node_list) if node1 != node2}
print(dist)


# 找出最小子圈
def subtour(edges):
    unvisited = node_list[:]
    cycle = node_list[:]
    while unvisited:  # 非空判断
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            # 找圈过程：
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(thiscycle) <= len(cycle):
            cycle = thiscycle  # New shortest sub tour
    return cycle


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a_s list of edges selected in the solution_y
        vals = model.cbGetSolution(model._vars)
        selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if len(tour) < len(node_list):
            # add subtour elimination constr. for every pair of cities in subtour
            model.cbLazy(quicksum(model._vars[i, j] for i, j in combinations(tour, 2)) <= len(tour) - 1)


# 主模型
m = Model()

# 设置变量
vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

# 约束1：流平衡
for i, j in vars.keys():
    m.addConstr(vars[j, i] == vars[i, j])

# 约束2：出入度加和为2
m.addConstrs(vars.sum(c, '*') == 2 for c in node_list)

# 求解
m._vars = vars
m.Params.lazyConstraints = 1    # 设置该参数才能进行回调求解
m.optimize(subtourelim)
print('\n Best Value:', m.objVal)