"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : gurobi_test.py
@File : BD_II2_Inout.py
@Author : Hsh
@Time : 2022-11-04 12:13

"""

from gurobipy import *
import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import time


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance

def set_parameters(data, I, J):
    loc1 = list(data['需求地'])
    loc1 = loc1[0:I]
    loc2 = list(data['库存地'])
    loc2 = loc2[0:J]
    loc_I = [loc1[i].split(',') for i in range(I)]
    loc_I = sum(loc_I, [])
    loc_I = np.array(loc_I, dtype=float)
    loc_I = loc_I.reshape([I, 2])
    loc_J = [loc2[j].split(',') for j in range(J)]
    loc_J = sum(loc_J, [])
    loc_J = np.array(loc_J, dtype=float)
    loc_J = loc_J.reshape([J, 2])

    c = []
    for i in range(I):
        lng1 = loc_I[i][0]
        lat1 = loc_I[i][1]
        for j in range(J):
            lng2 = loc_J[j][0]
            lat2 = loc_J[j][1]
            dis = geodistance(lng1, lat1, lng2, lat2)
            c.append(dis)
    c = np.array(c, dtype='float')
    c = c.reshape([I, J])

    # sets
    n = list(data['人口数'])
    n = n[0:I]
    d = n
    d = [round(d[i]) for i in range(I)]

    mo = list(data['房价'])
    mo = mo[0:J]
    f = [round(mo[j] / 10000) for j in range(J)]

    return c, d, f


def d_cons():
    d_hat = np.zeros((I, S))  # 按照场景的随机分布

    # 计算需求的随机分布
    np.random.seed(a)
    a1 = np.random.uniform(1 - u, 1 + u, size=S)  # 均匀分布
    for s in range(S):
        for i in range(I):
            d_hat[i, s] = a1[s] * d[i]

    return d_hat


def my_lazy_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        tss = time.time()
        sp_time = 0

        sol = model.cbGetSolution(model._vars)
        temp_n = sol[0:J].copy()
        hat_theta_list = sol[J:].copy()

        dual_lazy_obj = {}
        new_optimal_a = {}
        new_optimal_d = {}
        new_optimal_e = {}

        for s in range(S):
            if s == 0:
                sp = Model()
                sp.Params.outputflag = 0  # 不输出日志文件
                sp.Params.Method = 1
                # dual variables
                cb_b = {}
                cb_c = {}
                cb_d = {}
                cb_e = {}
                cb_a = sp.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='cb_a')
                for i in range(I):
                    cb_e[i] = sp.addVar(lb=0,
                                        ub=GRB.INFINITY,
                                        vtype=GRB.CONTINUOUS,
                                        name='cb_e')
                    for k in range(i + 1, I):
                        cb_b[i, k] = sp.addVar(lb=0,
                                               ub=GRB.INFINITY,
                                               vtype=GRB.CONTINUOUS,
                                               name='cb_b'+str(i)+str(k))
                        cb_c[i, k] = sp.addVar(lb=0,
                                               ub=GRB.INFINITY,
                                               vtype=GRB.CONTINUOUS,
                                               name='cb_c'+str(i)+str(k))
                for j in range(J):
                    cb_d[j] = sp.addVar(vtype=GRB.CONTINUOUS,
                                        name='cb_d')
                sp.update()
                # set dual objective
                obj = quicksum(temp_n[j] * cb_d[j] for j in range(J)) - quicksum(d_hat[i, s] * cb_e[i] for i in range(I)) - G * cb_a
                sp.setObjective(obj, GRB.MAXIMIZE)

                # # Add constrains
                # dual constrain 1
                for i in range(I):
                    for j in range(J):
                        sp.addConstr((cb_d[j] - cb_e[i] - c[i][j] + quicksum(
                            (cb_b[i, k] - cb_c[i, k]) / d_hat[i, s] for k in range(i + 1, I)) + quicksum(
                            (cb_c[k, i] - cb_b[k, i]) / d_hat[i, s] for k in range(I) if k < i)) <= 0,
                                     name='cb_dual_constrains_1'+str(i)+str(j))
                # dual constrain 2
                for i in range(I):
                    for k in range(i + 1, I):
                        sp.addConstr(cb_b[i, k] + cb_c[i, k] <= cb_a,
                                     name='cb_dual_constrains_2')
                sp.update()
            else:
                # # update constrain
                # getConstrByName : 根据名字获得约束
                # setAttr('RNS', constr) : 设置相关的Attr
                # chgCoeff(constr, Var, value) : 改变约束中的变量系数
                for i in range(I):
                    for j in range(J):
                        dual_cons_name = 'cb_dual_constrains_1' + str(i) + str(j)
                        const_2 = sp.getConstrByName(dual_cons_name)
                        for k in range(i + 1, I):
                            cb_b_var1 = sp.getVarByName('cb_b' + str(i) + str(k))
                            cb_c_var1 = sp.getVarByName('cb_c' + str(i) + str(k))
                            sp.chgCoeff(const_2, cb_b_var1, (1 / d_hat[i, s]))
                            sp.chgCoeff(const_2, cb_c_var1, (-1 / d_hat[i, s]))
                        for k in range(I):
                            if k < i:
                                cb_b_var2 = sp.getVarByName('cb_b' + str(k) + str(i))
                                cb_c_var2 = sp.getVarByName('cb_c' + str(k) + str(i))
                                sp.chgCoeff(const_2, cb_b_var2, (-1 / d_hat[i, s]))
                                sp.chgCoeff(const_2, cb_c_var2, (1 / d_hat[i, s]))
                sp.update()

            sp.optimize()
            # add MP constrains
            if sp.status == GRB.Status.OPTIMAL:
                dual_lazy_obj[s] = sp.objVal
                new_optimal_a[s] = cb_a.x
                new_optimal_d[s] = sp.getAttr('X', cb_d)
                new_optimal_e[s] = sp.getAttr('X', cb_e)
                sp_time += sp.runtime
            else:
                print('No Solution !!')

        sum_lazy_obj_dict = {}
        begin_num = 0
        for cur_theta in range(0, theta_num):
            sum_lazy_obj = 0

            end_num = begin_num + s_part
            if cur_theta == theta_num - 1:
                end_num = S
            for s in range(begin_num, end_num):
                sum_lazy_obj += (1/s_part)*dual_lazy_obj[s]
            sum_lazy_obj_dict[cur_theta] = sum_lazy_obj
            begin_num = begin_num + s_part

        begin_num = 0
        for cur_theta in range(0, theta_num):
            if (hat_theta_list[cur_theta]) < sum_lazy_obj_dict[cur_theta]:
                lazy_theta = LinExpr()

                end_num = begin_num + s_part
                if cur_theta == theta_num - 1:
                    end_num = S

                for s in range(begin_num, end_num):
                    lazy_theta_s = LinExpr()
                    for j in range(J):
                        lazy_theta_s += (model._vars[j] * new_optimal_d[s][j])
                    for i in range(I):
                        lazy_theta_s += (-d_hat[i, s] * new_optimal_e[s][i])
                    lazy_theta_s += (-G * new_optimal_a[s])

                    lazy_theta += (1/S)*lazy_theta_s

                model.cbLazy(((s_part/S)*model._vars[J+cur_theta] >= lazy_theta))
            begin_num = begin_num + s_part

        tee = time.time()
        this_lazy = tee-tss
        global all_lazy_time
        all_lazy_time += this_lazy
        global sp_lazy_time
        sp_lazy_time += sp_time


def Benders():
    # Model
    MP = Model()  # 主问题模型

    # primal variables
    N = MP.addVars(J, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='N')
    z = MP.addVars(theta_num, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z')

    obj = quicksum(f[j] * N[j] for j in range(J)) + quicksum((s_part/S) * z[cur] for cur in range(theta_num))
    MP.setObjective(obj, GRB.MINIMIZE)

    # #  constrains
    # primal constrain
    MP.addConstr(quicksum(N[j] for j in range(J)) == K)
    MP.update()

    # initial data
    main_ub = GRB.INFINITY
    main_lb = -GRB.INFINITY
    benders_gap = GRB.INFINITY
    t_root = 0
    iter_num = 0
    bad_bound = 0
    bad_gap = 0

    apply_N = {}
    sp_obj_dic = {}
    sp_A = {}
    sp_D = {}
    sp_E = {}

    # in_out strategy
    lamda = 0.2
    delta = 0.00002
    N_bar = {j: 1 for j in range(J)}
    init_N = {j: 1 for j in range(J)}
    mp_N = {}
    apply_sp_obj = {}


    while True:
        iter_num += 1

        MP.optimize()
        # MP.write('mp.lp')
        t_root += MP.runtime

        if MP.status == GRB.OPTIMAL:
            apply_N = MP.getAttr('X', N)
            mp_obj = MP.objVal
        else:
            print('No Solution !!!')

        for j in range(J):
            N_bar[j] = (N_bar[j] + apply_N[j]) / 2
            mp_N[j] = lamda * apply_N[j] + (1 - lamda) * N_bar[j] + delta * init_N[j]

        # dual model
        for s in range(S):
            if s == 0:
                SP = Model()
                SP.Params.outputflag = 0  # 不输出日志文件
                SP.Params.Method = 1
                # dual variables
                B = {}
                C = {}
                D = {}
                E = {}
                A = SP.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='A')
                for i in range(I):
                    E[i] = SP.addVar(lb=0,
                                     ub=GRB.INFINITY,
                                     vtype=GRB.CONTINUOUS,
                                     name='E')
                    for k in range(i + 1, I):
                        B[i, k] = SP.addVar(lb=0,
                                            ub=GRB.INFINITY,
                                            vtype=GRB.CONTINUOUS,
                                            name='B'+str(i)+str(k))
                        C[i, k] = SP.addVar(lb=0,
                                            ub=GRB.INFINITY,
                                            vtype=GRB.CONTINUOUS,
                                            name='C'+str(i)+str(k))
                for j in range(J):
                    D[j] = SP.addVar(vtype=GRB.CONTINUOUS,
                                     name='D')
                SP.update()
                # set dual objective
                obj = quicksum(mp_N[j] * D[j] for j in range(J)) - quicksum(d_hat[i, s] * E[i] for i in range(I)) - G * A
                SP.setObjective(obj, GRB.MAXIMIZE)

                # # Add constrains
                # dual constrain 1
                for i in range(I):
                    for j in range(J):
                        SP.addConstr((D[j] - E[i] - c[i][j] + quicksum(
                            (B[i, k] - C[i, k]) / d_hat[i, s] for k in range(i + 1, I)) + quicksum(
                            (C[k, i] - B[k, i]) / d_hat[i, s] for k in range(I) if k < i)) <= 0,
                                     name='dual_constrains_1'+str(i)+str(j))
                # dual constrain 2
                for i in range(I):
                    for k in range(i + 1, I):
                        SP.addConstr(B[i, k] + C[i, k] <= A,
                                     name='dual_constrains_2')
                SP.update()
            else:
                # update objective
                new_obj = quicksum(apply_N[j] * D[j] for j in range(J)) - quicksum(
                    d_hat[i, s] * E[i] for i in range(I)) - G * A
                SP.setObjective(new_obj, GRB.MAXIMIZE)

                # # update constrain
                # getConstrByName : 根据名字获得约束
                # setAttr('RNS', constr) : 设置相关的Attr
                # chgCoeff(constr, Var, value) : 改变约束中的变量系数
                for i in range(I):
                    for j in range(J):
                        cons_name = 'dual_constrains_1' + str(i) + str(j)
                        const_2 = SP.getConstrByName(cons_name)
                        for k in range(i + 1, I):
                            b_var1 = SP.getVarByName('B' + str(i) + str(k))
                            c_var1 = SP.getVarByName('C' + str(i) + str(k))
                            SP.chgCoeff(const_2, b_var1, (1 / d_hat[i, s]))
                            SP.chgCoeff(const_2, c_var1, (-1 / d_hat[i, s]))
                        for k in range(I):
                            if k < i:
                                b_var2 = SP.getVarByName('B' + str(k) + str(i))
                                c_var2 = SP.getVarByName('C' + str(k) + str(i))
                                SP.chgCoeff(const_2, b_var2, (-1 / d_hat[i, s]))
                                SP.chgCoeff(const_2, c_var2, (1 / d_hat[i, s]))
                SP.update()

            SP.optimize()
            # add MP constrains
            if SP.status == GRB.Status.OPTIMAL:
                sp_obj_dic[s] = SP.objVal
                sp_A[s] = A.x
                sp_D[s] = SP.getAttr('X', D)
                sp_E[s] = SP.getAttr('X', E)
            else:
                print('No Solution !!!')

        UB = sum(apply_N[j] * f[j] for j in range(J)) + (1/S)*sum(sp_obj_dic[s] for s in range(S))
        LB = mp_obj
        if UB >= main_ub:
            bad_bound += 1
        else:
            bad_bound = 0
        main_ub = min(main_ub, UB)

        temp_gap = benders_gap
        benders_gap = (main_ub - LB) / main_ub
        if benders_gap >= temp_gap:
            bad_gap += 1
        else:
            bad_gap = 0

        print('!!!!!')
        print('main_ub:', main_ub)
        print('main_lb:', LB)
        print('benders_gap:', benders_gap)
        print('!!!!!')

        if (benders_gap <= 0.005) or (bad_gap == 5):
            break

        if bad_bound == 5:
            lamda = 1
        elif bad_bound == 10:
            delta = 0

        if iter_num%5 == 0:
            for cons_i in MP.getConstrs():
                if cons_i.slack > 0:
                    MP.remove(cons_i)

        # 向主问题添加约束
        all_vars = MP.getVars()
        z_list = all_vars[J:].copy()

        con_sp_obj_dict = {}
        begin_num = 0
        for cur_theta in range(0, theta_num):
            sum_dual_obj = 0

            end_num = begin_num + s_part
            if cur_theta == theta_num - 1:
                end_num = S
            for s in range(begin_num, end_num):
                sum_dual_obj += (1/s_part)*sp_obj_dic[s]
            con_sp_obj_dict[cur_theta] = sum_dual_obj
            begin_num = begin_num+s_part

        begin_num = 0
        for cur_theta in range(0, theta_num):
            theta_dual_obj = con_sp_obj_dict[cur_theta]
            if (z_list[cur_theta].getAttr('X')) < theta_dual_obj:
                benders_theta = LinExpr()

                end_num = begin_num + s_part
                if cur_theta == theta_num - 1:
                    end_num = S

                for s in range(begin_num, end_num):
                    benders_theta_s = LinExpr()
                    for j in range(J):
                        benders_theta_s += (N[j] * sp_D[s][j])
                    for i in range(I):
                        benders_theta_s += (-d_hat[i, s] * sp_E[s][i])
                    benders_theta_s += (-G * sp_A[s])
                    benders_theta += (1/S)*benders_theta_s
                    benders_theta_s.clear()

                MP.addConstr(((s_part/S)*z[cur_theta] >= benders_theta))
            begin_num = begin_num + s_part

        MP.update()

    root_obj = min(LB, main_ub)

    del_num = 0
    for cons_ii in MP.getConstrs():
        #print(cons_ii.ConstrName, cons_ii.slack)
        if cons_ii.slack > 0:
            del_num += 1
            MP.remove(cons_ii)
    MP.update()

    print('迭代次数：', iter_num)
    print('松弛约束数量：', del_num)

    for j in range(J):
        N_name = 'N[' + str(j) + ']'
        N = MP.getVarByName(N_name)
        N.setAttr('VType', 'I')

    MP.update()
    MP._vars = MP.getVars()
    MP.Params.lazyConstraints = 1
    MP.Params.MIPGap = 0.005
    MP.optimize(my_lazy_callback)

    print('\n')
    print('hhhhh', all_lazy_time, sp_lazy_time)

    extra_time = all_lazy_time - sp_lazy_time
    print('mp.runtime: ', MP.runtime)
    print('多余时间: ', extra_time)

    t_lazy = MP.runtime - extra_time
    print(t_root, t_lazy)

    myruntime = t_root + t_lazy
    #root_obj, myruntime, t_root, t_lazy, iter_num, MP.mipgap, MP.getAttr('X', MP.getVars())

    return MP.objVal, myruntime


if __name__ == '__main__':
    # 参数设置
    S_list = [100, 200, 400]
    output_value = []
    for S in S_list:
        if S == 100 or S == 200:
            theta_list = [100]
        else:
            theta_list = [200]
        for theta_num in theta_list:
                    I = 25  # 需求点
                    J = 10  # 供给点
                    # S = 400  # 场景数
                    # theta_num = 100     # 整合组数
                    s_part = S//theta_num
                    K = 8000   # 物资总数
                    G = 3  # 基尼系数
                    u = 0.774596669    # 波动系数
                    data = pd.read_excel('15.39.xlsx')
                    c, d, f = set_parameters(data, I, J)
                    time_list = []
                    for th in range(1, 6):
                        a = th
                        d_hat = d_cons()
                        all_lazy_time = 0
                        sp_lazy_time = 0
                        mp, t = Benders()
                        print('obj:', mp)
                        print('time:', t)
                        time_list.append(t)
                    fast_time = min(time_list)
                    average_time = sum(time_list) / 5
                    print('\n')
                    print('!!!!')
                    print('S:', S, 'V:', theta_num)
                    print('BD_fast_time:', fast_time)
                    print('BD_average_time:', average_time)
                    print('\n')
                    output_value.append([I, J, S, theta_num, fast_time, average_time])
    output_value = np.array(output_value)
    data_df = pd.DataFrame(output_value)  # 将ndarray格式转换为DataFrame

    # 更改表的索引
    data_df.columns = ['I', 'J', 'S', 'V', 'BD_Fast', 'BD_Average']  # 列索引

    # 写入excel表格
    writer = pd.ExcelWriter('output2.xlsx')  # 创建名称为homework2的excel表格
    data_df.to_excel(writer)  # float_format 控制精度
    writer.save()