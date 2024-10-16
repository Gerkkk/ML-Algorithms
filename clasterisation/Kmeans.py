import pandas as pd
import numpy as np

class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_ = None
        self.inertia_ = None


    def __str__(self):
        ret = 'MyKMeans class: n_clusters='
        ret += str(self.n_clusters)
        ret += ', max_iter='
        ret += str(self.max_iter)
        ret += ', n_init='
        ret += str(self.n_init)
        ret += ', random_state='
        ret += str(self.random_state)
        return ret

    def iterate_algorithm(self, X):
        centroids = []

        for u in range(self.n_clusters):
            cur_centroid = []
            X_col = X.columns

            for i in range(X.shape[1]):
                cur = X[X_col[i]]
                cur_max, cur_min = max(cur), min(cur)

                cur_centroid.append(np.random.uniform(cur_min, cur_max))

            centroids.append(tuple(cur_centroid))


        for k in range(self.max_iter):
            clusters = [[] for i in range(self.n_clusters)]
            for i in range(X.shape[0]):
                point = X.iloc[[i]]
                point = np.array(point)

                answer = None
                dist = -1

                for u in range(len(centroids)):
                    cur_dist = np.sqrt(np.sum((point - centroids[u]) ** 2))
                    if (answer == None) or (cur_dist < dist):
                        dist = cur_dist
                        answer = u
                clusters[answer].append(point)

            f1 = 0
            for i in range(len(centroids)):
                if len(clusters[i]) > 0:
                    f1 = 1
                    centroids[i] = np.sum(clusters[i], axis=0)[0] / len(clusters[i])

            if f1 == 0:
                break

        return centroids

    def fit(self, X):
        np.random.seed(seed=self.random_state)

        self.cluster_centers_ = None
        self.inertia_ = None

        for j in range(self.n_init):

            cur_centroids = self.iterate_algorithm(X)

            clusters_sum = [0 for i in range(self.n_clusters)]

            for i in range(X.shape[0]):
                point = np.array(X.iloc[[i]])

                answer = None
                dist = -1

                for u in range(len(cur_centroids)):
                    cur_dist = np.sqrt(np.sum((point - cur_centroids[u]) ** 2))
                    if (answer == None) or (cur_dist < dist):
                        dist = cur_dist
                        answer = u

                clusters_sum[answer] += (dist ** 2)

            cur_WCSS = sum(clusters_sum)


            if (self.cluster_centers_ == None) or (cur_WCSS < self.inertia_):
                self.cluster_centers_ = tuple(cur_centroids)
                self.inertia_ = cur_WCSS

    def predict(self, X):
        ans = []
        for i in range(X.shape[0]):
            point = np.array(X.iloc[[i]])

            answer = None
            dist = -1

            for u in range(len(self.cluster_centers_)):
                cur_dist = np.sqrt(np.sum((point - self.cluster_centers_[u]) ** 2))
                if (answer == None) or (cur_dist < dist):
                    dist = cur_dist
                    answer = u

            ans.append(answer + 1)

        return ans

