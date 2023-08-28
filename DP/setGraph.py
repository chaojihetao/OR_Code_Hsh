"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : Dijkstra.py
@File : setGraph.py
@Author : Hsh
@Time : 2023-02-05 22:18

"""
import networkx as nx
from matplotlib import pyplot as plt


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
    isDraw = 1  # 0：不画图；1：画图
    maxStage = 4  # 最大阶段数
    A = {('v1', 'v2'): 2,
         ('v1', 'v3'): 5,
         ('v2', 'v3'): 1,
         ('v2', 'v4'): 4,
         ('v3', 'v2'): 3,
         ('v3', 'v4'): 1}
    
    Graph = setGraph(A, maxStage, origin_node, isDraw)
