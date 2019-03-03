import numpy as np
np.random.seed(1337)  
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras import optimizers
import matplotlib.pyplot as plt

X = np.linspace(-10, 10, 2000)
np.random.shuffle(X)    # 将数据集随机化
Y = 0.5 * np.power(X,2) + np.random.normal(0, 0.05, (2000, )) # 假设我们真实模型为：Y=0.5X+2
plt.scatter(X,Y)
plt.show()

X_train, Y_train = X[:1600], Y[:1600]     # 把前160个数据放到训练集
X_test,  Y_test  = X[1600:], Y[1600:]       # 把后40个点放到测试集

model = Sequential () # Keras有两种类型的模型，序贯模型（Sequential）和函数式模型
model.add(Dense(output_dim=10, input_dim=1)) # 通过add()方法一层层添加模型
model.add(Activation('softmax'))
model.add(Dense(output_dim=1, input_dim=5)) # 通过add()方法一层层添加模型
model.compile(loss='mse', optimizer='sgd')

print('Training -----------')
for step in range(3001):
    cost = model.train_on_batch(X_train, Y_train) # Keras有很多开始训练的函数，这里用train_on_batch（）
    if step % 100 == 0:
        print('train cost: ', cost)

print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()    # 查看训练出的网络参数
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.hold()
plt.scatter(X_test, Y_test)
plt.scatter(X_test, Y_pred, c='r')
plt.show()
