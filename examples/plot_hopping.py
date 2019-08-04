import matplotlib.pyplot as plt
import numpy as np

data1 = np.genfromtxt('avoided_crossing.txt')
data2 = np.genfromtxt('dualavoided.txt')

plt.figure(1)
plt.plot(data1[:, 0], data1[:, 1], label='GroundReflection')
plt.plot(data1[:, 0], data1[:, 3], label='GroundTransmission')
plt.plot(data1[:, 0], data1[:, 4], label='ExcitedTransmission')
plt.legend()

plt.figure(2)
p = data2[:, 0]
ke = p **2 / 4000
x = np.log(ke)
plt.plot(x, data2[:, 1], label='GroundReflection')
plt.plot(x, data2[:, 3], label='GroundTransmission')
plt.plot(x, data2[:, 4], label='ExcitedTransmission')
plt.legend()

data3 = np.genfromtxt('extended_coupling.txt')
plt.figure(3)
plt.plot(data3[:, 0], data3[:, 1], label='GroundReflection')
plt.plot(data3[:, 0], data3[:, 3], label='GroundTransmission')
plt.plot(data3[:, 0], data3[:, 2], label='ExcitedReflection')
plt.legend()

plt.show()