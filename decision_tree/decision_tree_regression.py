import pandas as pd
import numpy as np


def calc_MSE(pre_s):
    if len(pre_s) == 0:
        return 0
    s = [i[1] for i in pre_s]
    mean_s = np.array([np.mean(s)] * len(s))
    return np.sum(np.power(mean_s - np.array(s), 2)) / len(s)

def get_best_split(X, y, dividers):
    Real_X = X.join(pd.DataFrame(y))

    best_col = None
    best_split = None
    best_ig = None

    for cur_column in X.columns:
        cur = pd.DataFrame(Real_X[cur_column]).join(pd.DataFrame(y))
        cur = cur.to_numpy()

        mass = []
        for i in cur:
            mass.append([i[0], i[1]])

        mass.sort()

        s0 = calc_MSE(mass)

        j = 0
        i = 0
        while j < len(dividers[cur_column]):

            while i + 1 < len(mass) and mass[i + 1][0] <= dividers[cur_column][j]:
                i += 1
            l, r = mass[:i + 1], mass[i + 1:]

            cur_ig = s0 - calc_MSE(l) * ((i + 1) / len(mass)) - calc_MSE(r) * ((len(mass) - i - 1) / len(mass))

            if best_col == None or cur_ig > best_ig:
                best_ig = cur_ig
                best_split = (mass[i][0] + mass[i + 1][0]) / 2
                best_col = cur_column
            j += 1

    return best_col, best_split, best_ig


class Tree_node:
    def __init__(self, parent=None, depth=0):
        self.type = None
        self.info = None
        self.parent = parent
        self.depth = depth
        self.left_child = None
        self.right_child = None

        self.split = None

class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

        self.leafs_cnt = 1
        self.head = Tree_node(depth=0)
        self.bins = bins
        self.dividers = None

    def __str__(self):
        ret = 'MyTreeReg class: max_depth='
        ret += str(self.max_depth)
        ret += ', min_samples_split='
        ret += str(self.min_samples_split)
        ret += ', max_leafs='
        ret += str(self.max_leafs)
        return ret


    def get_dividers(self, X):
        ret = dict()

        for cur_column in X.columns:
            cur = list(X[cur_column])
            cur.sort()
            if self.bins is None or len(set(cur)) <= self.bins:
                ans = []
                for i in range(1, len(cur)):
                    ans.append((cur[i - 1] + cur[i])/2)
                ret[cur_column] = ans
            else:

                d = np.histogram(cur, self.bins)[1]
                d = d[1:-1]
                ret[cur_column] = d
        self.dividers = ret

    def leaf_wise_build(self, vertex, X, y):
        if X.shape[0] < self.min_samples_split or vertex.depth + 1 > self.max_depth or\
                (self.leafs_cnt + 1 > self.max_leafs and self.leafs_cnt > 1):
            vertex.type = 'leaf'

            vertex.split = np.mean(y)
            if vertex.parent:
                vertex.info = ' ' * (4 * (vertex.depth)) + f'leaf_{str("left") if vertex == vertex.parent.left_child else str("right")} = {vertex.split}'
            else:
                vertex.info = ' ' * (4 * (vertex.depth)) + f'leaf_left = {vertex.split}'

        else:
            self.leafs_cnt += 1
            vertex.type = 'node'

            col, split, ig = get_best_split(X, y, self.dividers)

            if split >= min(X[col]) and split < max(X[col]):
                vertex.split = (col, split, ig)
                X['target_'] = y

                Real_X1, Real_X2 = X[X[col] <= split], X[X[col] > split]

                mask = [True for i in range(X.shape[1])]
                mask[-1] = False

                X1, y1 = Real_X1.loc[:, mask], Real_X1['target_']
                X2, y2 = Real_X2.loc[:, mask], Real_X2['target_']

                left_son = Tree_node(vertex, vertex.depth + 1)
                self.leaf_wise_build(left_son, X1, y1)
                vertex.left_child = left_son

                right_son = Tree_node(vertex, vertex.depth + 1)
                self.leaf_wise_build(right_son, X2, y2)
                vertex.right_child = right_son

                vertex.info = ' ' * (4 * vertex.depth) + f'{col} > {split}'
            else:
                vertex.type = 'leaf'

                vertex.split = np.mean(y)
                if vertex.parent:
                    vertex.info = ' ' * (4 * (
                        vertex.depth)) + f'leaf_{str("left") if vertex == vertex.parent.left_child else str("right")} = {vertex.split}'
                else:
                    vertex.info = ' ' * (4 * (vertex.depth)) + f'leaf_left = {vertex.split}'

    def fit(self, X, y):
        self.get_dividers(X)
        self.leaf_wise_build(self.head, X, y)


    def print_tree_tech(self, vertex):
        if vertex.parent is None:
            print(vertex.info)
        if vertex.type == 'node':
            print(vertex.left_child.info)
            self.print_tree_tech(vertex.left_child)
            print(vertex.right_child.info)
            self.print_tree_tech(vertex.right_child)

    def print_tree(self):
        self.print_tree_tech(self.head)

    def vertex_predict(self, vertex, x):
        if vertex.type == 'node':
            if x[vertex.split[0]] <= vertex.split[1]:
                return self.vertex_predict(vertex.left_child, x)
            else:
                return self.vertex_predict(vertex.right_child, x)
        else:
            return vertex.split


    def predict(self, X):
        ans = []
        for index, row in X.iterrows():
            prediction = self.vertex_predict(self.head, row)
            ans.append(prediction)

        return ans