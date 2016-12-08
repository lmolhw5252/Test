from PIL import Image
import  numpy as np
import matplotlib.pyplot as plt
'''获取所有图片，并reshape成10x784的大小'''
img_all = []
for i in range(10):
    url='%s.png'%i
    img = Image.open(url).convert('L')
    img = np.array(img.resize((28,28)))
    img = np.reshape(img,(1,-1))
    img_all = np.array(list(img_all)+list(img))
# print(img_all.shape)
y_test = np.eye(10,dtype=float)
# print(y_test)

for i in range(10):
    # 创建一个两行四列的图，图的数量为index+1
    plt.subplot(2, 5, i+1)
    # 设置轴属性,忽略横轴
    plt.axis('off')
    plt.imshow(img_all[i].reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
    print(img_all[i].reshape((28,28)))
    plt.title('Prediction: %i' % i)
plt.show()


# pil_im = Image.open('1.png').convert('L')
# plt.imshow(pil_im,cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
#
# out = np.array(pil_im.resize((8,8)))
#
# plt.imshow(out,cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
# print(out.shape)
# out = np.reshape(out,(1,-1))
# print(out.shape)
