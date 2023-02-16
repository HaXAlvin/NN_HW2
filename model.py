import numpy as np
from matplotlib import pyplot as plt
from numpy import random
random.seed(1234)

class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for _ in range(numCenters)]
        self.beta = 0.3
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, center, data_point):
        # print(data_point)
        assert len(data_point) == self.indim, f"{len(data_point)},{self.indim}"
        return np.exp(-self.beta * np.linalg.norm(center-data_point)**2)
     
    def _calcAct(self, datas):
        # calculate activations of RBFs
        matrix = np.zeros((datas.shape[0], self.numCenters), float)
        for ci, center in enumerate(self.centers):
            for di, data in enumerate(datas):
                matrix[di,ci] = self._basisfunc(center, data)
        return matrix
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
        # self.centers = [[a,0,0] for a in range(0,40)] + \
        #                [[0,a,0] for a in range(0,40)] + \
        #                [[0,0,a] for a in range(0,40)]
         
        # calculate activations of RBFs
        G = self._calcAct(X)
         
        # calculate output weights (pseudoinverse)
        self.W = np.dot(np.linalg.pinv(G), Y)
        # print(self.W)
         
    def predict(self, X):
        """ X: matrix of dimensions n x indim """
        if isinstance(X,list):
            X = np.array(X)
            X = np.expand_dims(X,axis=0)
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y

# if __name__ == '__main__':
#     data = np.loadtxt("./train4dAll.txt")

#     # rbf regression
#     rbf = RBF(3, 500, 1)
#     # for _ in range(5):
#     rbf.train(data[:,:3], data[:,3]+1e-1)
#     z = rbf.predict(data[:,:3])
       
#     # plot original data
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(1,2,1, projection='3d')
#     sc = ax.scatter(data[:,0],data[:,1],data[:,2], c=data[:,3], marker='o')
#     fig.colorbar(sc)
#     ax.set_xlabel('x axis')
#     ax.set_ylabel('y axis')
#     ax.set_zlabel('z axis')
#     # plot trained data
#     ax = fig.add_subplot(1,2,2, projection='3d')
#     sc = ax.scatter(data[:,0],data[:,1],data[:,2], c=z, marker='o')
#     fig.colorbar(sc)
#     ax.set_xlabel('x axis')
#     ax.set_ylabel('y axis')
#     ax.set_zlabel('z axis')
#     plt.show()

if __name__ == "__main__":
    data = np.loadtxt("./train4dAll.txt")

    # rbf regression
    rbf = RBF(2, 500, 1)
    # data[:,4] = data[:,2]-data[:1]
    # print(np.stack([data[:,0],data[:,2]-data[:,1]]).T)
    new_data = np.stack([data[:,0],data[:,2]-data[:,1]]).T
    rbf.train(new_data, data[:,3]+1e-1)
    z = rbf.predict(new_data)

    # plot original data
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(3,3,1)
    sc = ax.scatter(new_data[:,0],new_data[:,1], c=data[:,3], marker='o')
    fig.colorbar(sc)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    # plot trained data
    ax = fig.add_subplot(3,3,2)
    sc = ax.scatter(new_data[:,0],new_data[:,1], c=z, marker='o')
    fig.colorbar(sc)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')

    ax = fig.add_subplot(3,3,4)
    ax.scatter(data[:,0],data[:,3])

    ax = fig.add_subplot(3,3,7)
    ax.scatter(data[:,1],data[:,3])

    ax = fig.add_subplot(3,3,5)
    mean = np.mean(rbf.W)
    std = np.std(rbf.W)
    zs = (rbf.W-mean)/std
    new_zs = np.where(np.abs(zs)<1,zs,0)
    new_W = new_zs*std+mean
    ax.scatter(list(range(500)),new_W)
    ax = fig.add_subplot(3,3,8)
    ax.scatter(list(range(500)),new_zs)

    ax = fig.add_subplot(3,3,6)
    ax.scatter(list(range(500)),rbf.W)
    ax = fig.add_subplot(3,3,9)
    ax.scatter(list(range(500)),zs)
    
    # ax.scatter(data[:,2]-data[:,1],data[:,3])

    ax = fig.add_subplot(3,3,3)
    diff = data[:,2]-data[:,1]
    mean = np.mean(diff)
    std = np.std(diff)
    zs = (diff-mean)/std
    ax.scatter(zs,data[:,3])
    ax.plot([2,-2],[-40,40],c='r')
    ax.plot([3,-3],[-40,40],c='g')

    plt.show()
    # np.mean()
    # while(True):
    #     x1 = float(input("x1: "))
    #     x2 = float(input("x2: "))
    #     x3 = float(input("x3: "))
    #     print("Predict:")
    #     print(rbf.predict([x1,x2,x3]))
    #     print("Next")
