import numpy as np
import random
from PIL import Image
from math import floor
from math import ceil
from math import sqrt

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
z_buffer = np.full((1000, 1000), np.inf)
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


def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2, img_mat):
    denumerator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denumerator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denumerator
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def triangle(x0, y0, x1, y1, x2, y2, z0, z1, z2, img_mat, color, z_buffer):
    xmin = floor(min(x0, x1, x2))
    xmax = ceil(max(x0, x1, x2))
    ymin = floor(min(y0, y1, y2))
    ymax = ceil(max(y0, y1, y2))
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > 999: xmax = 999
    if ymax > 999: ymax = 999

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            l0, l1, l2 = barycentric_coordinates(x + 0.5, y + 0.5, x0, y0, x1, y1, x2, y2, img_mat)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                z = l0 * z0 + l1 * z1 + l2 * z2
                ## отрисовываем то, что ближе к нам по z
                if z < z_buffer[x, y]:
                    img_mat[x, y] = color      
                    z_buffer[x, y] = z          

def normal(v0, v1, v2):
    # вектор из v1 в v2
    ax = v1[0] - v2[0]
    ay = v1[1] - v2[1]
    az = v1[2] - v2[2]

    # вектор из v1 в v0
    bx = v1[0] - v0[0]
    by = v1[1] - v0[1]
    bz = v1[2] - v0[2]

    # координаты нормали
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    
    length = sqrt(nx**2 + ny**2 + nz**2) # длина вектора
    return nx / length, ny / length, nz / length # нормировка

for j in range(len(f)):
    i0 = int(f[j][0]) - 1
    i1 = int(f[j][1]) - 1
    i2 = int(f[j][2]) - 1

    v0 = v[i0]
    v1 = v[i1]
    v2 = v[i2]

    nx, ny, nz = normal(v0, v1, v2)

    if nz >= 0: continue ## нормаль от нас

    brightness = int(-255 * nz) 
    color = (brightness, 0, 0)

    x0 = -v0[0] * 7000 + 500
    y0 = -v0[1] * 7000 + 750
    x1 = -v1[0] * 7000 + 500
    y1 = -v1[1] * 7000 + 750
    x2 = -v2[0] * 7000 + 500
    y2 = -v2[1] * 7000 + 750

    triangle(y0, x0, y1, x1, y2, x2, v0[2], v1[2], v2[2], img_mat, color, z_buffer)

img = Image.fromarray(img_mat)
img.save('img.png')
              

