import numpy as np
from matplotlib import pyplot as plt
import os

os.system('cls')

accData = np.load(r'Data\accelData.npy')
acc_x, acc_y, acc_z = [], [], []

for i in range(len(accData[0])):
    acc_x.append(accData[0][i][0])
    acc_y.append(accData[0][i][1])
    acc_z.append(accData[0][i][2])

plt.plot(acc_x, label="X")
plt.plot(acc_y, label="Y")
plt.plot(acc_z, label="Z")
plt.legend()
plt.title("Accelerometer data")
plt.xlabel("Samples")
plt.ylabel("Acceleration (mG)")
plt.show()
