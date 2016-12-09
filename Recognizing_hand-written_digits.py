# print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()
#1797张8x8的图
# print(digits.images.shape)

#显示每张图
# for i in range(10):
#     plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('digits.target[%s]' %i)
#     plt.show()


# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

#把图像数组和target放在一起
images_and_labels = list(zip(digits.images, digits.target))

# print(images_and_labels[0])

# for index, (image, label) in enumerate(images_and_labels[:4]):
#     #创建一个两行四列的图，图的数量为index+1
#     plt.subplot(2, 4, index + 1)
#     #设置轴属性,忽略横轴
#     plt.axis('off')
#     #在axes上显示图像
#     #X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4)
#     #camp,如果为None，则忽略图像的RGB值
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#
#     plt.title('Training: %i' % label)
# plt.show()

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:

n_samples = len(digits.images)
# print("n_samples",n_samples)

#把图像变成1797x64的矩阵
data = digits.images.reshape((n_samples, -1))
# print(data.shape)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
'''使用前一半进行训练'''
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
# 用后一半进行测试
expected = digits.target[n_samples / 2:]
# print(data[n_samples / 2:].shape)

predicted = classifier.predict(data[n_samples / 2:])

'''全部作为训练集'''
# classifier.fit(data,digits.target)
# print(predicted.shape)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
#
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# print(predicted.__class__)

# print(digits.images[n_samples / 2:].shape)
images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))

# print(digits.images[n_samples / 2:].shape)

# for index, (image, prediction) in enumerate(images_and_predictions[5:9]):
#     # print(prediction)
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     # print(image.shape)
#     plt.title('Prediction: %i' % prediction)
#
# plt.show()

'''输入图片测试'''
test_im = Image.open('9.png').convert('L')
out = np.array(test_im.resize((8,8)))
out_test = np.reshape(out,(1,-1))



'''使用自带数据集测试'''
# test_im = digits.data[3]
# out = np.reshape(test_im,(8,-1))
# # print(test_im.shape)
# out_test = np.reshape(out,(1,-1))


'''测试数据'''
test  = classifier.predict(out_test)
plt.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('Prediction: %i' %test)
plt.show()
