print(__doc__)

import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

#设置数据的位置
mnist = fetch_mldata("MNIST original",data_home='/home/lmolhw/下载/scikit_learn_data')

# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
# print(X.__sizeof__())
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]



# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)

#构建一个多层感知机分类器
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
# #拟合训练集，返回训练的MLP模型
mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

#把面板设置成4行4列的,axes是一个数组
fig, axes = plt.subplots(4, 4)
# fig.show()
# print(axes)

print(mlp.coefs_[1].size)

# use global min / max to ensure all weights are shown on the same scale

vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()

# print(vmin,vmax)
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(39, 20), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())
    # print(coef.size)
plt.show()
