import pandas as pd
import numpy as np

class MyLogReg:
    def __init__(self,  weights=None, n_iter=10, learning_rate=0.1):
        self.eps = 1e-15

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        ret = 'MyLogReg class: '
        ret += 'n_iter=' + str(self.n_iter) + ', '
        ret += 'learning_rate=' + str(self.learning_rate)
        return ret

    @staticmethod
    def _mae(y_true: np.array, y_pred: np.array):
        return (y_true - y_pred).abs().mean()

    def fit(self, X, y, verbose=False):
        arr_X = np.array(X.values)
        arr_Y = np.array(y.values)

        #adding column of 1s
        column = np.array([1.] * arr_X.shape[0])
        arr_X = np.vstack((column, arr_X.T)).T

        num_of_features = arr_X.shape[1]


        real_weights = np.array([1.0] * num_of_features)
        self.weights = real_weights

        for i in range(self.n_iter):
            z = arr_X @ self.weights
            predict = 1 / (1 + np.exp(-z))

            LogLoss = (-1.0) * (1/arr_X.shape[0]) * ((np.log(predict) * arr_Y) + ((1 - arr_Y) * np.log(1 - arr_Y)) )

            grad = ((predict - arr_Y) @ arr_X) / arr_X.shape[0]

            self.weights -= self.learning_rate * grad

            #printing log
            if verbose and i % verbose == 0:
                print(f'{i}| loss: {LogLoss})')

    def get_coef(self):
        ans = self.weights[1:]
        return ans

    def predict_proba(self, X):
        arr_X = np.array(X.values)

        column = np.array([1.] * arr_X.shape[0])
        arr_X = np.vstack((column, arr_X.T)).T

        z = arr_X @ self.weights
        predict = 1 / (1 + np.exp(-z))
        return predict.mean()

    def predict(self, X):
        arr_X = np.array(X.values)

        column = np.array([1.] * arr_X.shape[0])
        arr_X = np.vstack((column, arr_X.T)).T

        z = arr_X @ self.weights

        predict = 1 / (1 + np.exp(-z))
        ans = np.array([1 if predict[i] > 0.5 else 0 for i in range(predict.shape[0])])
        return sum(ans)

