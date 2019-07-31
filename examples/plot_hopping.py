import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('avoided_crossing.txt')

plt.plot(data[:, 0], data[:, 1], label='GroundReflection')
plt.plot(data[:, 0], data[:, 3], label='GroundTransmission')
plt.plot(data[:, 0], data[:, 4], label='ExcitedTransmission')

plt.legend()
plt.show()
