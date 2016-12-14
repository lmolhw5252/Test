from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
sess = tf.InteractiveSession()
# img = Image.open('lena.jpg')
# img = np.array(img)
# if img.ndim == 3:
#     img = img[:,:,0]
# plt.subplot(221)
# plt.imshow(img)
# plt.subplot(222)
# plt.imshow(img, cmap ='gray')
# plt.subplot(223)
# plt.imshow(img, cmap = plt.cm.gray)
# plt.subplot(224)
# plt.imshow(img, cmap = plt.cm.gray_r)
# plt.show()
'''获取所有图片，并reshape成10x784的大小'''
img_all = []
for i in range(10):
    url='%s.png'%i
    img = Image.open(url).convert('L')
    img = np.array(img.resize((28,28)))
    img = np.reshape(img,(1,-1))
    img_all = np.array(list(img_all)+list(img))
pre = np.eye(10,dtype=float)

# pre = sess.run(y_conv,feed_dict={x:img_all})
for i in range(10):
    for j in range(10):
        if pre[i][j]==1:
            # 创建一个两行四列的图，图的数量为index+1
            plt.subplot(2, 5, i + 1)
            # 设置轴属性,忽略横轴
            plt.axis('off')
            plt.imshow(img_all[i].reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Prediction: %i' % j)
            # print("test accuracy %g" % accuracy.eval(feed_dict={
            #     x:img_all,y_:y_test,keep_prob:1.0}))
plt.show()