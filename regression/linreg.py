import pandas as pd
import numpy as np
import random

class MyLineReg:
    def __init__(self,  weights=None, n_iter=100, learning_rate=0.1,
                 metric=None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric

        self.sgd_sample = sgd_sample
        self.random_state = random_state

        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.last_X = None
        self.last_Y = None


    def __str__(self):
        ret = 'MyLineReg class: '
        ret += 'n_iter=' + str(self.n_iter) + ', '
        ret += 'learning_rate=' + str(self.learning_rate)
        return ret

    @staticmethod
    def _reg_none(x_true: np.array, y_true: np.array, y_pred: np.array):
        MSE = ((y_true - y_pred) ** 2).mean()
        grad = (((y_pred - y_true) @ x_true) * 2) / x_true.shape[0]
        return MSE, grad

    @staticmethod
    def _reg_l1(x_true: np.array, y_true: np.array, y_pred: np.array, weights: np.array, l1_coef: float):
        MSE = ((y_true - y_pred) ** 2).mean() + l1_coef * sum(abs(weights))
        grad = ( (((y_pred - y_true) @ x_true) * 2) / x_true.shape[0] ) + l1_coef * np.sign(weights)
        return MSE, grad

    @staticmethod
    def _reg_l2(x_true: np.array, y_true: np.array, y_pred: np.array, weights: np.array, l2_coef: float):
        MSE = ((y_true - y_pred) ** 2).mean() + l2_coef * (weights ** 2).sum()
        grad = ((((y_pred - y_true) @ x_true) * 2) / x_true.shape[0]) + 2 * l2_coef * weights
        return MSE, grad

    @staticmethod
    def _reg_ElasticNet(x_true: np.array, y_true: np.array, y_pred: np.array, weights: np.array, l1_coef: float, l2_coef: float):
        MSE = ((y_true - y_pred) ** 2).mean() + l1_coef * sum(abs(weights)) + l2_coef * (weights ** 2).sum()
        grad = ((((y_pred - y_true) @ x_true) * 2) / x_true.shape[0]) + l1_coef * np.sign(weights) + 2 * l2_coef * weights
        return MSE, grad

    def fit(self, X, y, verbose=False):

        seed = random.seed(self.random_state)

        arr_X = np.array(X.values)
        arr_Y = np.array(y.values)

        #adding column of 1s
        column = np.array([1.] * arr_X.shape[0])
        arr_X = np.vstack((column, arr_X.T)).T

        num_of_features = arr_X.shape[1]

        real_weights = np.array([1.] * num_of_features)
        self.weights = real_weights

        for i in range(1, self.n_iter + 1):
            if self.sgd_sample:
                if type(self.sgd_sample) == float:
                    self.sgd_sample = round(self.sgd_sample * arr_X.shape[0])
                sample_rows_idx = random.sample(range(arr_X.shape[0]), self.sgd_sample)

                real_arr_X = arr_X[sample_rows_idx]
                real_arr_Y = arr_Y[sample_rows_idx]
            else:
                real_arr_X = arr_X
                real_arr_Y = arr_Y

            predict = real_arr_X @ self.weights

            if self.reg == None:
                MSE, grad = self._reg_none(real_arr_X, real_arr_Y, predict)
            elif self.reg == 'l1':
                MSE, grad = self._reg_l1(real_arr_X, real_arr_Y, predict, self.weights, self.l1_coef)
            elif self.reg == 'l2':
                MSE, grad = self._reg_l2(real_arr_X, real_arr_Y, predict, self.weights, self.l2_coef)
            elif self.reg == 'elasticnet':
                MSE, grad = self._reg_ElasticNet(real_arr_X, real_arr_Y, predict, self.weights, self.l1_coef, self.l2_coef)

            if type(self.learning_rate) == float:
                real_lr = self.learning_rate
            else:
                real_lr = self.learning_rate(i)

            self.weights -= real_lr * grad

            #printing log
            if verbose and i % verbose == 0:
                y_mean = np.array([np.mean(real_arr_Y)] * len(y))
                print(f'{i})')

            self.last_X = real_arr_X
            self.last_Y = real_arr_Y

    def get_coef(self):
        ans = self.weights[1:]
        return ans

    def predict(self, X):
        arr_X = np.array(X.values)

        # adding column of 1s
        column = np.array([1.] * arr_X.shape[0])
        arr_X = np.vstack((column, arr_X.T)).T

        predict = arr_X @ self.weights

        return sum(predict)

    def get_best_score(self):
            predict = self.last_X @ self.weights

            y_mean = np.array([np.mean(self.last_Y)] * len(self.last_Y))

            ans = 0
            if self.metric == 'mae':
                ans = (abs(predict - self.last_Y)).sum() / len(self.last_Y)
            elif self.metric == 'mse':
                ans = ((predict - self.last_Y) ** 2).sum() / len(self.last_Y)
            elif self.metric == 'rmse':
                ans = (((predict - self.last_Y) ** 2).sum() / len(self.last_Y)) ** 0.5
            elif self.metric == 'mape':
                ans = ((abs((self.last_Y - predict) / self.last_Y)).sum() * 100) / len(self.last_Y)
            elif self.metric == 'r2':
                ans = 1 - (((predict - self.last_Y) ** 2).sum() / (((self.last_Y - y_mean) ** 2).sum()))

            return ans

