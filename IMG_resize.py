from PIL import Image
import  numpy as np
import matplotlib.pyplot as plt

pil_im = Image.open('00004.png').convert('L')

out = np.array(pil_im.resize((64,64)))

plt.imshow(out,cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
print(out.__class__)