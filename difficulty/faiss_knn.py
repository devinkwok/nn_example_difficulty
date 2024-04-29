import numpy as np
import faiss

# adapted from https://gist.githubusercontent.com/j-adamczyk/74ee808ffd53cd8545a49f185a908584/raw/3bf67a7eead909008f3ecaffdaf046805d0a1243/knn_with_faiss.py
# and https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
class FaissKNeighbors:
    
    def __init__(self, k=30, device="cuda"):
        self.index = None
        self.y = None
        self.k = k
        self.device = device

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(X)
        self.y = y

    def predict(self, X):
        _, indices = self.index.search(X, k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
