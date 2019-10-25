import cv2
import numpy as np

#img_size = (200,200, 3)
#img = np.ones(img_size) * 255
img = cv2.imread('/home/weihao/Downloads/IMG_1134.JPG')
img_size = img.shape

# polar equation
theta = np.linspace(0, np.pi, 1000)
r = 1 / (np.sin(theta) - np.cos(theta))

# polar to cartesian
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

x,y = polar2cart(r, theta)
x1, x2, y1, y2 = x[0], x[1], y[0], y[1]

# line equation y = f(X)
def line_eq(X):
    m = (y2 - y1) / (x2 - x1)
    return m * (X - x1) + y1

line = np.vectorize(line_eq)

x = np.arange(0, img_size[0])
y = line(x).astype(np.uint)

print( (x[0], y[0]), (x[-1], y[-1]))
print(img.shape)
cv2.line(img, (x[0], y[0]), (x[-1], y[-1]), (0,0,255))
cv2.imshow("foo",img)
cv2.waitKey()