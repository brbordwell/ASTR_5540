#!/usr/bin/python3.4


######Importing##Useful##Packages###################################
import numpy as np
from numpy.fft import fft as fft
from matplotlib import pyplot as plt
from copy import deepcopy
import time
import pyfits
from os import sys
import pdb
from time import time as time


def mmt_matrix(x, w = 1, wavenumber=True):
    """Purpose: To generate the transform matrix for the MMT

       Input: 
       x = the dependent values
       w = quadratic weights related to the dependent values
           default: Fourier series on a uniform grid, w_i = 1
       wavenumber = an option to also return the wavenumber
                    default: return wavenumber

       Output:
       mmt = the transform matrix
       """

    N = len(x)
    it = np.arange(N).reshape((1,N))
    A = np.exp(1j * (-2*np.pi*np.dot(it.T,it)/(N)))

    if wavenumber:
        delta_x = np.diff(x)[0]
        k = (np.arange(N)/N-0.5)/delta_x
        k = np.fft.ifftshift(k)
        return A, k
    else:
        return A


def mmt_fft_compare(x, fx, timing=False):
    f = fx(x)
    MMT, k = mmt_matrix(x)

    c_mmt = lambda k: np.dot(k[0], k[1])
    if timing: 
        c_mmt, t_mmt = timer(c_mmt, MMT, f)
    else: 
        c_mmt = c_mmt((MMT,f))
    ps_mmt = np.abs(c_mmt)**2


    c_fft = lambda k: fft(k[0])
    if timing: 
        c_fft, t_fft = timer(c_fft, f)
    else: 
        c_fft = c_fft((f,))
    ps_fft = np.abs(c_fft)**2
    
    if timing:
        return t_mmt, t_fft
    else:
        return c_mmt, ps_mmt, c_fft, ps_fft, k



def timer(f, *args):
    init = time()  
    res = f(args)
    fin = time()
    t0 = np.array(fin-init)

    init = time()  
    res = f(args)
    fin = time()
    t1 = np.append(t0,fin-init)

    while np.mean(t1)-np.mean(t0) > .01*np.mean(t0):
        t0 = t1
        init = time()  
        res = f(args)
        fin = time()
        t1 = np.append(t0,fin-init)

    return res, np.mean(t1)




if __name__=="__main__":
    #Generating and plotting the Lorentzian
    N = int(sys.argv[1])
    a = float(sys.argv[2])
    x_vals = lambda x: np.linspace(-np.pi, np.pi, num=x, endpoint=False)
    Lorentzian = lambda x: a**2 / (x**2 + a**2)
    
    x = x_vals(N)
    f = Lorentzian(x)

    plt.plot(x,f)
    plt.title("Lorentzian (a=0.1, x=[-$\pi$,$\pi$) )")
    plt.xlim(min(x),max(x))
    plt.savefig("hw8_fig1.png")
    plt.clf()

    #Obtaining power spectra from both FFT and MMT
    c_mmt, ps_mmt, c_fft, ps_fft, k = mmt_fft_compare(x, Lorentzian)


    #Plotting and comparing
    fig, axes = plt.subplots(3, sharex=True)
    
    axes[0].semilogy(k, ps_fft, 'b.')
    axes[0].set_ylabel("Coefficient Power")
    axes[0].set_title("FFT Power Spectrum")
    axes[0].set_xlim(min(k),max(k))

    axes[1].semilogy(k, ps_mmt,'b.')
    axes[1].set_ylabel("Coefficient Power")
    axes[1].set_title("MMT Power Spectrum")
    axes[1].set_xlim(min(k),max(k))

    rel_err = np.abs(c_fft-c_mmt)/np.abs(c_fft)
    axes[2].semilogy(k,rel_err,'b.')
    axes[2].set_ylabel("|FFT-MMT|/|FFT|")
    axes[2].set_title("Relative error in coefficients")
    axes[2].set_xlim(min(k),max(k))
    axes[2].set_xlabel("Wavenumber")

    fig.savefig("hw8_fig2.png")
    fig.clf()


    #Performing the inverse MMT and plotting
    f_immt = np.dot(mmt_matrix(x,wavenumber=False).T,c_mmt)/N

    fig, axes = plt.subplots(2, sharex=True)

    axes[0].plot(x,f, label="Original")
    axes[0].plot(x, f_immt+.5, label="Inverted MMT+ 0.5")
    axes[0].set_title("Lorentzian (a=0.1, x=[-$\pi$,$\pi$) )")
    axes[0].legend(loc="upper right")

    rel_err = np.abs(f-f_immt)/np.abs(f)
    axes[1].semilogy(x, rel_err)
    axes[1].set_ylabel("|f(x)-f$_{IMMT}$(x)|/f(x)")
    axes[1].set_title("Relative error in f$_{IMMT}$(x)")
    axes[1].set_ylim(min(rel_err),max(rel_err))
    axes[1].set_xlim(min(x),max(x))
    fig.savefig("hw8_fig3.png")
    fig.clf()


    #Investigating competitiveness of transforms with N
    times_mmt = []
    times_fft = []
    N_arr = 2**3*2**np.arange(10)
    for n in N_arr:
        t_mmt, t_fft = mmt_fft_compare(x_vals(n), Lorentzian, timing=True)
        times_mmt.append(t_mmt)
        times_fft.append(t_fft)

    plt.semilogy(N_arr, times_mmt, label="MMT")
    plt.semilogy(N_arr, times_fft, label="FFT")
    plt.legend(loc="upper left")
    plt.xlabel("N, number of data elements")
    plt.ylabel("Time to transform [s]")
    plt.title("MMT vs. FFT: Time")
    plt.ylim(min([min(times_mmt),min(times_fft)]), max([max(times_mmt),max(times_fft)]))
    plt.xlim(min(N_arr), max(N_arr))
    plt.savefig("hw8_fig4.png")

    
    #Investigating results with N != 2**n
    print("For N = 2048:")
    print("\t MMT:", times_mmt[-2])
    print("\t FFT:", times_fft[-2])
    print("\t MMT/FFT:", times_mmt[-2]/times_fft[-2])
    print()  ; print()

    t_mmt, t_fft = mmt_fft_compare(x_vals(2049), Lorentzian, timing=True)
    print("For N = 2049:")
    print("\t MMT:", t_mmt)
    print("\t FFT:", t_fft)
    print("\t MMT/FFT:", t_mmt/t_fft)
