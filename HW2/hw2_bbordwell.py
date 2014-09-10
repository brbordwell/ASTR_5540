#!/usr/bin/python3.4

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

A = np.array([[-.9880, 1.800, -0.9793, -0.5977, -.7819],
              [-1.9417, -0.5835, -0.1846, -0.7250, 1.0422],
              [0.6003, -0.0287, -0.5446, -2.0667, -0.3961],
              [0.8222, 1.4453, 1.3369, -0.6069, 0.8043], 
              [-0.4187, -0.2939, 1.4814, -0.2119, -1.2771]])

evals = la.eigvals(A)
print(evals)

mnr, mxr = min(evals.real), max(evals.real)
mni, mxi = min(evals.imag), max(evals.imag)

plt.plot(evals.real, evals.imag, 'bo', label= 'Eigenvalues')
plt.plot([mnr-1, mxr+1], [0,0], 'g--', label='$I$ = 0')
plt.plot([0,0],[mni-1, mxi+1], 'k--', label='$R$ = 0')
x1,x2,y1,y2 = plt.axis()
plt.axis((mnr-.5, mxr+.5, mni-.5, mxi+.5))

plt.xlabel('Real Component')
plt.ylabel('Imaginary Component')
plt.title('Distribution of Eigenvalues')
plt.legend(loc='center')
plt.savefig('hw2.png')
plt.show()
