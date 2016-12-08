from PIL import Image
import  numpy as np
import matplotlib.pyplot as plt

pil_im = Image.open('1.png').convert('L')
plt.imshow(pil_im,cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

out = np.array(pil_im.resize((8,8)))

plt.imshow(out,cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
print(out.shape)
out = np.reshape(out,(1,-1))
print(out.shape)
