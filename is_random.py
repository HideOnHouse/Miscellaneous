import random

from matplotlib import pyplot as plt

size = 10000
x = [random.random() for i in range(size)]
y = [random.random() for i in range(size)]
plt.figure(dpi=300)
plt.scatter(x, y, s=0.1)
plt.show()
