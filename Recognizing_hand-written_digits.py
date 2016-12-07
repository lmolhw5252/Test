<<<<<<< Updated upstream
# print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt


=======
import matplotlib.pyplot as plt

>>>>>>> Stashed changes
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()
<<<<<<< Updated upstream
# print(digits.data[0])
# for i in range(10):
#     plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('digits.target[%s]' %i)
#     plt.show()
=======

>>>>>>> Stashed changes
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

images_and_labels = list(zip(digits.images, digits.target))

<<<<<<< Updated upstream
# print(images_and_labels)

for index, (image, label) in enumerate(images_and_labels[:4]):
    #创建一个两行四列的图，图的数量为index+1
    plt.subplot(2, 4, index + 1)
    #设置轴属性,turns off the axis lines and labels
    plt.axis('off')
    #在axes上显示图像
    #X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4)
    #camp,如果为None，则忽略图像的RGB值
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('Training: %i' % label)
    # plt.show()
=======
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
>>>>>>> Stashed changes

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
<<<<<<< Updated upstream
# print(n_samples)

data = digits.images.reshape((n_samples, -1))
# print(data)
=======
data = digits.images.reshape((n_samples, -1))
>>>>>>> Stashed changes

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
<<<<<<< Updated upstream
# 使用前一半进行训练
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
# 用后一半进行测试
expected = digits.target[n_samples / 2:]

=======
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
>>>>>>> Stashed changes
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

<<<<<<< Updated upstream

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))

=======
images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
>>>>>>> Stashed changes
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()