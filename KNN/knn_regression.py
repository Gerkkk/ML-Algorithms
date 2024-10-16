import pandas as pd
import numpy as np


class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight

        train_size = None
        arr_X = None
        arr_Y = None

    def __str__(self):
        ret = 'MyKNNReg class: k='
        ret += str(self.k)
        return ret

    def fit(self, X, y):
        self.arr_X = np.array(X.values)
        self.arr_Y = np.array(y.values)
        self.train_size = (self.arr_X.shape[0], self.arr_X.shape[1])

    @staticmethod
    def d_eucl(self, known_point, pred_point):
        return np.sqrt(np.sum((known_point - pred_point) ** 2))

    @staticmethod
    def d_cheb(self, known_point, pred_point):
        return np.max(np.absolute(known_point - pred_point))

    @staticmethod
    def d_manh(self, known_point, pred_point):
        return np.sum(np.absolute(known_point - pred_point))

    @staticmethod
    def d_cos(self, known_point, pred_point):
        abs_known = np.sqrt(np.sum((known_point ** 2)))
        abs_pred = np.sqrt(np.sum((pred_point ** 2)))
        return 1 - np.sum(known_point * pred_point) / (abs_known * abs_pred)

    def weight_uniform(self, nearest):
        types = [nearest[i][1] for i in range(self.k)]
        ans = sum(types) / len(types)
        return ans

    def weight_rank(self, nearest):
        types = []

        koef = sum([1/(i+1) for i in range(len(nearest))])

        for i in range(self.k):
            types.append(nearest[i][1] / (koef * (i + 1)))

        types = np.array(types)
        nums = np.array([nearest[i][1] for i in range(self.k)])
        ans = types * nums
        return ans

    def weight_distance(self, nearest):
        types = []

        koef = sum([(1 / nearest[i][0]) for i in range(self.k)])

        for i in range(self.k):
            types.append(nearest[i][1] / (koef * nearest[i][0]))

        types = np.array(types)
        nums = np.array([nearest[i][1] for i in range(self.k)])
        ans = types * nums
        return ans

    def predict(self, X):
        cur_X = np.array(X.values)

        predictions = []
        for pred_point in cur_X:
            nearest = []

            for known_point, answer in zip(self.arr_X, self.arr_Y):
                if self.metric == 'euclidean':
                    cur_dist = self.d_eucl(self, known_point, pred_point)
                elif self.metric == 'chebyshev':
                    cur_dist = self.d_cheb(self, known_point, pred_point)
                elif self.metric == 'manhattan':
                    cur_dist = self.d_manh(self, known_point, pred_point)
                elif self.metric == 'cosine':
                    cur_dist = self.d_cos(self, known_point, pred_point)
                else:
                    cur_dist = -1
                nearest.append((cur_dist, answer))

            nearest.sort()

            if self.weight == 'uniform':
                ans = self.weight_uniform(nearest)
            elif self.weight == 'rank':
                ans = self.weight_rank(nearest)
            elif self.weight == 'distance':
                ans = self.weight_distance(nearest)


            predictions.append(ans)

        return np.array(predictions)
