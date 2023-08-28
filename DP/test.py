"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : Dijkstra.py
@File : MTZ_new.py
@Author : Hsh
@Time : 2023-02-06 9:31

"""
import matplotlib.pyplot as plt
import networkx as nx


def Dijkstra(Graph, origin, target):
    # 读取图对象数据
    Node_list = []
    for node in Graph.nodes:
        Node_list.append(node)
        if node == origin:
            Graph.nodes[node]['min_dis'] = 0
        else:
            Graph.nodes[node]['min_dis'] = 9999999  # big_M

    while len(Node_list) > 0:
        current_node = None
        minPathLength = 9999999  # big_M

        # 找寻根节点
        for node in Node_list:
            if Graph.nodes[node]['min_dis'] < minPathLength:
                current_node = node
                minPathLength = Graph.nodes[node]['min_dis']

        if current_node is not None:
            Node_list.remove(current_node)

        for neighbor in Graph.successors(current_node):  # 遍历根节点的相邻节点
            arc_key = (current_node, neighbor)
            distance = Graph.nodes[current_node]['min_dis'] + Graph.edges[arc_key]['length']
            if distance < Graph.nodes[neighbor]['min_dis']:
                Graph.nodes[neighbor]['min_dis'] = distance  # 更新路径距离
                Graph.nodes[neighbor]['previous_node'] = current_node  # 更新前向节点

    # 输出路径及其长度
    opt_dis = Graph.nodes[target]['min_dis']  # 最短路径长度
    opt_path = [current_node]  # 最短路径——回溯前向节点
    while current_node != origin:
        current_node = Graph.nodes[current_node]['previous_node']
        opt_path.insert(0, current_node)

    return opt_dis, opt_path


def setGraph(A, maxStage, origin, isDraw):
    Node_pos = {}  # 坐标
    Graph = nx.DiGraph()  # 建立空有向图
    current_list = [origin]  # 当前节点列表
    stage = 0  # 阶段

    while current_list:  # 后继节点不为空
        if stage == maxStage:
            break
        stage += 1  # 下一阶段
        number = -1  # 各阶段点顺序
        update_list = []  # 更新节点列表

        for node in current_list:
            # 起点
            if node == origin:
                Graph.add_node(str(stage - 1) + origin,
                               node=origin,
                               min_dis=0,
                               previous_node=None)
                Node_pos[str(stage - 1) + origin] = (0, 0)
            # 寻找后继节点
            for key in A.keys():
                if key[0] == node:
                    number += 1
                    if str(stage) + key[1] not in Node_pos.keys():  # 判断节点是否已存在
                        # 绘制下一阶段点
                        Graph.add_node(str(stage) + key[1],  # 点标识
                                       node=key[1],  # 实际节点标签
                                       min_dis=999999999,
                                       previous_node=None)
                        Node_pos[str(stage) + key[1]] = (stage, number)

                    # 绘制下一阶段弧
                    Graph.add_edge(str(stage - 1) + key[0], str(stage) + key[1],
                                   length=A[key[0], key[1]])

                    # 更新列表
                    update_list.append(key[1])
        current_list = update_list

    # 画图
    if isDraw == 1:
        pos = Node_pos  # pos相当于坐标系（具体用法可见draw函数）
        nx.draw(Graph, pos, with_labels=True, alpha=1)
        nx.draw_networkx_nodes(Graph, pos, node_size=400)
        edgeLabel = nx.get_edge_attributes(Graph, 'length')
        nx.draw_networkx_edge_labels(Graph, pos, edge_labels=edgeLabel)
        plt.show()

    return Graph


if __name__ == '__main__':
    origin_node = 'v1'
    isDraw = 0  # 0：不画图；1：画图
    maxStage = 8  # 最大阶段数
    A = {('v1', 'v2'): 2,
         ('v1', 'v6'): 3,
         ('v1', 'v4'): 1,
         ('v2', 'v5'): 5,
         ('v2', 'v3'): 6,
         ('v3', 'v8'): 6,
         ('v4', 'v2'): 10,
         ('v4', 'v7'): 2,
         ('v6', 'v7'): 4,
         ('v6', 'v4'): 5,
         ('v5', 'v3'): 9,
         ('v5', 'v8'): 4,
         ('v7', 'v5'): 3,
         ('v7', 'v2'): 7,
         ('v7', 'v8'): 8}

    Graph = setGraph(A, maxStage, origin_node, isDraw)
    origin = '0v1'
    target = '5v8'
    opt_dis, opt_path = Dijkstra(Graph, origin, target)
    print("最短路径长度: ", opt_dis)
    print("最短路径: ", opt_path)
