import numpy as np

X = np.load('GT.npy')

result = []
for row in X:
    result.append((row[0] + row[1] + row[2]) / 3)

print(result)
