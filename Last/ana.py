import numpy as np
from sklearn import linear_model, datasets

filename = "c:\\tmp\\r.txt"
data = np.loadtxt(filename)

print(data.shape)

p1 = data[:, 1:4]
p2 = data[:, 5:8]

print(p1.shape, p2.shape)

reg = linear_model.LinearRegression()
for a in range(3):
    reg.fit(p2, p1[:,a])
    print(reg.coef_, reg.intercept_, reg._residues)