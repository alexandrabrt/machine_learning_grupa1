import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("Figure_1.png")
plt.imshow(img)

plt.title("Imagine de exemplu")

plt.savefig('exemplu_salvat.png', format='png')
plt.savefig('exemplu_salvat.pdf', format='pdf')
plt.savefig('exemplu_salvat.svg', format='svg')
plt.savefig('exemplu_salvat.jpg', format='jpg')

plt.show()
