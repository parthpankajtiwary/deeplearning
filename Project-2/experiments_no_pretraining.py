from matplotlib import pyplot as plt
import numpy as np

from supervised_discriminator import train_discriminator as td

acc50 = td(.5)
acc25 = td(.25)
acc125 = td(.125)
acc625 = td(.0625)
acc3175 = td(.03175)
acc1 = td(1/128)

plt.title('Discriminator supervised training(features+dense)[No pre-training]')
plt.xlabel('epochs')
plt.ylabel('top-1 accuracy(%)')
plt.plot(list(np.arange(50,76)), [9]+acc50, '-b', label='50%')
plt.plot(list(np.arange(50,76)), [9]+acc25, '-g', label='25%')
plt.plot(list(np.arange(50,76)), [9]+acc125, '-r', label='12.5%')
plt.plot(list(np.arange(50,76)), [9]+acc625, '-y', label='6.25%')
plt.plot(list(np.arange(50,76)), [9]+acc3175, '-c', label='3.175%')
plt.plot(list(np.arange(50,76)), [9]+acc1, '-m', label='1/batch')
plt.legend()
plt.show()
