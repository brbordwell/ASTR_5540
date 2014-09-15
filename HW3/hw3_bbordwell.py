#!/usr/bin/python3.4

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from copy import deepcopy

###PLOT###GENERATION##########################################################

#Reading in data and obtaining indices of different objects
data = np.genfromtxt('mflux_lx_all.txt').transpose() #Col 1: Mx, Col2: Lx
qsun_inds = np.where(data[0] < 1e18) # Quiet Sun
XBP_inds = np.where((data[0] > 1e18) & (data[0] < 1e21)) # Xray bright pts
asun_inds = np.where((data[0] > 1e21) & (data[1] < 1e27)) # Active Sun
dwf_inds =  np.where((data[1] > 1e27) & (data[0] < 1e26)) # G,K,M dwarfs
TT_inds =  np.where(data[0] > 1e26) # T Tauri stars


#Plotting everything as in Fig1 of Pevtsov et al 2003
fig = plt.figure()  ;  ax = fig.add_subplot(1,1,1)
ax.loglog(data[0][qsun_inds],data[1][qsun_inds],'k.',label='Quiet Sun') # points
ax.loglog(data[0][XBP_inds],data[1][XBP_inds],'ks',label='X-ray Bright Points') # squares
ax.loglog(data[0][asun_inds],data[1][asun_inds],'kd',label='Solar Active Regions') # diamonds
ax.loglog(data[0][dwf_inds],data[1][dwf_inds],'kx',label='G,K,M Dwarfs') # Xs
ax.loglog(data[0][TT_inds],data[1][TT_inds],'ko',label='T Tauri Stars') # circles

#ax.legend(loc='lower right', fontsize= 8)
#fig.savefig('hw3_fig.png')
#plt.show()


##FITTING##COEFFICIENTS######################################################
A = np.array([np.log(data[0]),np.ones(len(data[0]))])
b = deepcopy(np.log(data[1]))

x = np.linalg.solve(np.dot(A,A.transpose()), np.dot(b,A.transpose()))
p = x[0]  ;  C = np.exp(x[1])
print('p = '+ str(p))
print('C = '+ str(C))

fit = np.dot(x,A)
avg_abs_err = np.mean(np.abs(b - fit))
print('Mean absolute error of fit is: '+str(avg_abs_err))


##PLOTTING##OUT##THE##FINAL##PLOT
ax.loglog(data[0],np.exp(fit),label='Least squares fit')

ax.set_xlabel('Magnetic Flux, M$_x$',fontsize=8)
ax.set_ylabel('$L_x$, erg s$^{-1}$', fontsize=8)
ax.set_title('X-ray spectral radiance vs. total unsigned magnetic flux for solar andstellar objects', fontsize=8)
ax.legend(loc='lower right', fontsize= 8)
ax.annotate('$\ln{L_x} = p\ln{\Phi}+\ln{C} = 1.15\ln{\Phi}+\ln{.873}$',xy=(5e20,5e23),xytext=(5e15,1e27), arrowprops=dict(facecolor='black',shrink=.05,width=2))
fig.savefig('hw3_fig.png')
plt.show()
