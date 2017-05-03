#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
from enum import Enum


class NodeTypes(Enum):
    """
    画图的节点类型配置
    """
    # 决策节点
    decision = dict(boxstyle='sawtooth', fc='0.8')
    # 叶子节点
    leaf = dict(boxstyle='round4', fc='0.8')
    # 箭头
    arrow = dict(arrowstyle='<-')


class Plotter(object):
    """
    画图的对象，画一个figure，一个subplot
    """

    def __init__(self):
        self._w = 0.0       # 画树的宽度
        self._d = 0.0       # 画树的高度
        self._xoff = 0.0    # 画树过程中，当前的x坐标
        self._yoff = 0.0     # 画树过程中，当前的y坐标
        self.__axes_prepared = False  # axes subplot是否已经存在
        self.prepare_figure()
        self.prepare_subplot()

    def prepare_figure(self):
        '''
        准备画布figure
        '''
        self.__figure = plt.figure(1, facecolor='white')

    def prepare_subplot(self, reprepare=False, **axprops):
        ''' 准备axes subplot
        Args:
            reprepare: 是否清空以前的重新准备一个subplot
            axprops: axprops=dict(xticks=[], yticks=[])指定x和y轴的属性，可以控制显示否
        '''
        # subplot axes 子图轴，一行一列No=1
        if reprepare is True:
            self.clear_figure()
        if not self.__axes_prepared:
            self.__axes = plt.subplot(111, frameon=False, **axprops)
            self.__axes_prepared = True

    def plot_annotation_node(self, point_pos, txt_pos, txt, node_type):
        """
        画一个带注释箭头的节点。point_pos --> txt_pos。在txt_pos画一个矩形，显示txt
        Args:
            point_pos: 点的位置(x, y)
            txt_pos: 文字的位置(x, y)
            txt: 注释的内容
            node_type: 矩形的类型，是descion或leaf
        Returns:
            None
        """
        self.__axes.annotate(txt, xy=point_pos, xytext=txt_pos, bbox=node_type,
                             arrowprops=NodeTypes.arrow.value, va='center', ha='center',
                             xycoords='axes fraction', textcoords='axes fraction')

    def show_image(self):
        """
        显示图片
        """
        plt.show()

    def clear_figure(self):
        '''
        清空图片
        '''
        self.__figure.clf()
        self.__axes_prepared = False

    def plot_txt_in_positions(self, p1, p2, txt):
        ''' 在两个点之间画上文字
        Args:
            p1: (x1, y1)
            p2: (x2, y2)
            txt: 要插入的内容
        '''
        x = (p1[0] + p2[0]) / 2
        y = (p1[1] + p2[1]) / 2
        self.__axes.text(x, y, txt, va='center', ha='center', rotation=30)

    def create_plot_tree(self, tree):
        ''' 展示一个树
        Args:
            tree: dict格式的树
        '''
        axprops = dict(xticks=[], yticks=[])
        self.prepare_subplot(True, **axprops)
        # 整颗树的叶子节点个数和深度
        self._w = float(self.get_leaves_num(tree))
        self._d = float(self.get_depth(tree))
        # 开始画树的坐标
        self._xoff = -0.5 / self._w
        self._yoff = 1.0
        self.plot_tree(tree, (0.5, 1.0), '')

    def plot_tree(self, tree, parent_pt, txt):
        ''' 画一棵树
        Args:
            tree: 当前需要画的树
            parent_pt: 父节点
            txt: 父节点与当前节点之间需要注释的内容，是父节点到当前节点的str(value)
        '''
        leaves_num = self.get_leaves_num(tree)
        depth = self.get_depth(tree)
        # 当前节点的文字内容
        cur_node_str = tree.keys()[0]   # 只有1个key
        # 开始画的位置
        x = self._xoff + (1.0 + float(leaves_num)) / 2.0 / self._w
        y = self._yoff
        cur_pt = (x, y)
        # 从parent_pt到cur_pt画出当前节点
        self.plot_annotation_node(
            parent_pt,
            cur_pt,
            cur_node_str,
            NodeTypes.decision.value)
        # 从parent_pt到cur_pt画分支信息，即str(value)
        self.plot_txt_in_positions(parent_pt, cur_pt, txt)

        # 画子树
        children = tree[cur_node_str]
        self._yoff -= 1.0 / self._d
        for key, value in children.items():
            if isinstance(value, dict):
                # 是树
                self.plot_tree(value, cur_pt, str(key))
            else:
                # 是叶子节点
                self._xoff += 1.0 / self._w
                child_pt = (self._xoff, self._yoff)
                self.plot_annotation_node(
                    cur_pt, child_pt, str(value), NodeTypes.leaf.value)
                self.plot_txt_in_positions(cur_pt, child_pt, str(key))
        self._yoff += 1.0 / self._d

    def retrieve_tree(self, i):
        """
        构造了n颗树
        Args:
            i: 第i颗树
        Returns:
            tree_list[i]: 第i颗树
        """
        tree_list = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {
                         0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                     ]
        return tree_list[i]

    def get_leaves_num(self, tree):
        """
        获得tree的叶子节点个数
        Args:
            tree: dict格式的tree
        Returns:
            leaves_num: 叶子节点个数
        """
        leaves_num = 0
        current_node = tree.keys()[0]  # 只有1个key
        children = tree[current_node]
        # print children
        for child_key in children.keys():
            if isinstance(children[child_key], dict):
                leaves_num += self.get_leaves_num(children[child_key])
            else:
                leaves_num += 1
        return leaves_num

    def get_depth(self, tree):
        """
        获得tree的深度
        Args:
            tree: dict格式的tree
        Returns:
            depth: 树的深度
        """
        depth = 0
        current_node = tree.keys()[0]
        children = tree[current_node]
        for key, value in children.items():
            c_depth = 1
            if isinstance(value, dict):
                c_depth += self.get_depth(value)
            if c_depth > depth:
                depth = c_depth
        return depth


def test_plotter_plot_node():
    '''
    测试plotter的绘图，画两个节点
    '''
    plotter = Plotter()
    plotter.prepare_subplot()
    plotter.plot_annotation_node(
        (0.1, 0.5), (0.5, 0.1), 'decision node', NodeTypes.decision.value)
    plotter.plot_annotation_node(
        (0.3, 0.8), (0.8, 0.1), 'leaf node', NodeTypes.leaf.value)
    plotter.show_image()


def test_plotter_plot_tree():
    '''测试画一颗树'''
    plotter = Plotter()
    tree = plotter.retrieve_tree(0)
    plotter.create_plot_tree(tree)
    plotter.show_image()


def main():
    # test_plotter_plot_node()
    test_plotter_plot_tree()


if __name__ == '__main__':
    main()
