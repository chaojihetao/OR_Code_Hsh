"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : CVRP_MTZ.py
@File : Dijkstra.py
@Author : Hsh
@Time : 2023-01-29 8:41

"""
import matplotlib.pyplot as plt
import networkx as nx

def setGraph(Node, A):
    # 构建图对象
    Graph = nx.DiGraph()

    # 点
    for name in Node.keys():
        Graph.add_node(name,
                       min_dis=0,
                       previous_node=None)

    # 弧
    for key in A.keys():
        Graph.add_edge(key[0], key[1],
                       length=A[key])

    # 画图
    pos = Node  # pos相当于坐标系（具体用法可见draw函数）
    nx.draw(Graph, pos, with_labels=True, alpha=1)
    edge_labels = nx.get_edge_attributes(Graph, 'length')
    nx.draw_networkx_edge_labels(Graph, pos, edge_labels=edge_labels)
    plt.show()

    return Graph


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


if __name__ == '__main__':
    # 导入图数据
    Node = {'v1': (0, 2),
            'v2': (2, 2),
            'v3': (4, 2),
            'v4': (1, 1),
            'v5': (3, 1),
            'v6': (0, 0),
            'v7': (2, 0),
            'v8': (4, 0)}

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

    # 构建图对象
    Graph = setGraph(Node, A)
    origin = 'v1'
    target = 'v8'
    opt_dis, opt_path = Dijkstra(Graph, origin, target)
    print("最短路径: ", opt_path)
    print("最短路径长度: ", opt_dis)

    # minPath = nx.dijkstra_path(Graph, source='1', target='6')  # 调用Networks内置函数求解
    # lenMinPath = nx.dijkstra_path_length(Graph, source='1', target='6')  # 最短路径长度
    # print("最短路径: ", minPath)
    # print("最短路径长度: ", lenMinPath)
