import numpy as np
from PIL import Image
from math import floor

def dda(x0, x1, y0, y1, img_mat):
    dmax = max(abs(floor(x0) - floor(x1)), abs(floor(y0) - floor(y1)))
    L = dmax + 1
    if(L== 1):
        img_mat[floor(x0), floor(y0)] = 255
        return
    dx = (x1 - x0) / (L - 1)
    dy = (y1 - y0) / (L - 1)
    for i in range(L):
        img_mat[floor(x0), floor(y0)] = 255
        x0 += dx
        y0 += dy

img_mat = np.zeros((1000,1000, 3), dtype=np.uint8)
v = []
f = []
file = open('model.obj')
for s in file:
    s1 = s.split()
    if(s1[0] == 'v'):
        v.append([float(s1[1]), float(s1[2]), float(s1[3])])
    if(s1[0] == 'f'):
        s2 = s1[1].split('/')
        s3 = s1[2].split('/')
        s4 = s1[3].split('/')
        f.append([float(s2[0]), float(s3[0]), float(s4[0])])

for j in range(len(f)):
    x0 = -v[int(f[j][0])-1][0] * 7000 + 500
    y0 = -v[int(f[j][0])-1][1] * 7000 + 750
    x1 = -v[int(f[j][1])-1][0] * 7000 + 500
    y1 = -v[int(f[j][1])-1][1] * 7000 + 750
    x2 = -v[int(f[j][2])-1][0] * 7000 + 500
    y2 = -v[int(f[j][2])-1][1] * 7000 + 750
    dda(y0, y1, x0, x1, img_mat)
    dda(y1, y2, x1, x2, img_mat)
    dda(y2, y0, x2, x0, img_mat)

img = Image.fromarray(img_mat)
img.save('img.png')
