import tensorflow as tf
import  input_data
import matplotlib as mlt
import numpy as np
from PIL import Image



mnist = input_data.read_data_sets('MNIST',one_hot=True)
print(mnist.__class__)

sess = tf.InteractiveSession()

#x是一个二维数组，第一个代表张数，第二个代表像素大小
x = tf.placeholder("float",shape=[None,784])
#y_也是一个二维数组，其中每一行为一个10维的one-hot向量，用于代表对应某一MNIST图片的类别
y_ = tf.placeholder("float",shape=[None,10])


#在模型中应该加入少量的噪声来打破对称性以及避免0梯度
#因为使用的是Relu神经元，所以可以使用一个较小的正数来初始化bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#vanilla版本
#卷积使用1步长（stride size），0边距（padding size）
#池化使用2x2的模板做max pooling

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#设置第一层卷积,由一个卷积接一个max pooling完成。卷积在每个5x5的pacth中算出32个特征
#卷积的权重张量形状是[5,5,1,32]，前两个维度是patch的大小，接着是输入的通道数目
#最后是输出的通道数目，而对于每一个输出通道都由一个对应的偏置量
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#把x变成4维向量2,3维代表宽高，最后一维代表图片的颜色通道数，rgb=3，灰度图=1
x_image = tf.reshape(x,[-1,28,28,1])

#我们把x_image和权值向量进行卷积，加上偏置项，然后应用Relu激活函数，最后进行max pooling
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积，每个5x5的patch会得到64个特征
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层,图片尺寸减小到7x7，我们加入一个由1024神经元的全连接层，用于处理整个图片。
#我们把池化层输入的张量reshape成一些向量，乘上权重矩阵，然后对其使用ReLU
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#为了减少过拟合，在输出层之前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#输出层softmax
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#测评

#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#步长
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess.run(tf.initialize_all_variables())

for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i%100 ==0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0],y_:batch[1],keep_prob:1.0
        })
        print("step %d, training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0
# }))
'''导入一张图片作为测试'''
test_im = Image.open('1.png').convert('L')
#
out = np.array(test_im.resize((28,28)))
# print(out.shape)

out_test = np.reshape(out,(1,-1))
out_test = tuple(out_test)
# print(out_test)
out_y = ([[0,1,0,0,0,0,0,0,0,0]])
print(out_y)

batch = mnist.train.next_batch(1)

print("test accuracy %g"% accuracy.eval(feed_dict={
    x:batch[0],y_:batch[1],keep_prob:1.0
}))



