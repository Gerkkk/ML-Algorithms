import pandas as pd
import numpy as np


class MyKNNClf:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight

        train_size = None
        arr_X = None
        arr_Y = None

    def __str__(self):
        ret = 'MyKNNClf class: k='
        ret += str(self.k)
        return ret

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

    def fit(self, X, y):
        self.arr_X = np.array(X.values)
        self.arr_Y = np.array(y.values)
        self.train_size = (self.arr_X.shape[0], self.arr_X.shape[1])

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
            types = [nearest[i][1] for i in range(self.k)]

            if self.weight == 'uniform':
                predictions.append(1 if types.count(1) >= types.count(0) else 0)
            elif self.weight == 'rank':
                denom = sum([1 / i for i in range(1, self.k + 1)])
                nom_0 = sum([((1 / i) if types[i - 1] == 0 else 0) for i in range(1, self.k + 1)])
                nom_1 = sum([((1 / i) if types[i - 1] == 1 else 0) for i in range(1, self.k + 1)])

                predictions.append(1 if (types.count(1) * nom_1) / denom >= (types.count(0) * nom_0) / denom else 0)
            elif self.weight == 'distance':
                denom = sum([1 / nearest[i][0] for i in range(self.k)])

                nom_0 = sum([((1 / nearest[i][0]) if types[i] == 0 else 0) for i in range(self.k)])
                nom_1 = sum([((1 / nearest[i][0]) if types[i] == 1 else 0) for i in range(self.k)])

                predictions.append(1 if (types.count(1) * nom_1) / denom >= (types.count(0) * nom_0) / denom else 0)


        return np.array(predictions)

    def predict_proba(self, X):
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

            types = [nearest[i][1] for i in range(self.k)]

            if self.weight == 'uniform':
                predictions.append(types.count(1) / len(types))
            elif self.weight == 'rank':
                denom = sum([1 / i for i in range(1, self.k + 1)])
                nom_1 = sum([((1 / i) if types[i - 1] == 1 else 0) for i in range(1, self.k + 1)])

                predictions.append(nom_1 / denom)
            elif self.weight == 'distance':
                denom = sum([1 / nearest[i][0] for i in range(self.k)])
                nom_1 = sum([((1 / nearest[i][0]) if types[i] == 1 else 0) for i in range(self.k)])

                predictions.append(nom_1 / denom)

        return np.array(predictions)

