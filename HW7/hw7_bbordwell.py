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
from copy import deepcopy


######Data##Reader##################################################
def read_data(file):
    """
    Purpose: Reads in and cleans up the Kepler light curve data.

    Input: 
    file = A string containing the filename to be read in

    Output:
    times = The times (in seconds) corresponding to the data points in the light curve.
    lightcurve = The "raw" lightcurve produced by the Kepler pipeline.
    PDC_lightcurve = The detrended lightcurve produced by the Kepler pipeline.
    """

    file = pyfits.open(file)
    data = file['LIGHTCURVE'].data
    raw_times = data['TIME']
    raw_lightcurve = data['SAP_FLUX']
    raw_PDC_lightcurve = data['PDCSAP_FLUX']

    # clean out NANS and infs in raw data stream
    # slightly more points are bad in PDC version
    good_data = np.isfinite(raw_PDC_lightcurve)
    lightcurve = raw_lightcurve[good_data]
    PDC_lightcurve = raw_PDC_lightcurve[good_data]
    times = raw_times[good_data]
    
    N_good_pts = len(lightcurve)
    N_bad_pts = len(raw_lightcurve)-N_good_pts
    print("{:d} good points and"
          " {:d} bad points in the lightcurve".format(N_good_pts, N_bad_pts))

    # note: the PDC_lightcurve is a corrected lightcurve from the Kepler 
    # data pipeline, that fixes some errors.
    # PDC means "Pre-Search Data Conditioning"
    return times, lightcurve, PDC_lightcurve



######A##function##to##solve##the##normal##equations#####################
def LS_fitter(x_data, y_data, N, QR=False):
    """
    Purpose: Performs a least squares polynomial fit of degree N to the data
    
    Input: 
    x_data, y_data = Independent and dependent data values 
    N = Degree of polynomial being fitted
    /QR = keyword option to solve the problem via a QR decomposition

    Output:
    fit = The fitted data.
    x = The coefficients of the polynomial fit
    abs_err = L2 norm for the fit
    time = The amount of time required to perform the fit, in seconds.
    """

    #Generating the A matrix
    x_data = (x_data-x_data.min())/(x_data.max()-x_data.min())
    A = np.array([x_data**i for i in range(N+1)])

    #Using the QR decomposition to avoid formation of ATA...
    init = time()
    if QR:
        q, r = np.linalg.qr(A.transpose())
        x = np.dot(np.linalg.inv(r),np.dot(q.transpose(),y_data))

    #Solving the problem with the normal equations route...
    else:
        x = np.linalg.solve(np.dot(A,A.transpose()), 
                            np.dot(y_data,A.transpose()).transpose())

    end = time()

    #Calculating the fit, residual, and L2 norm
    fit = np.dot(x.transpose(),A)  ;  res = (y_data-fit)**2

    if len(fit.shape) > 1: d = fit[0]  #acc. for weirdness from previous eds.
    else: d = fit

    abs_err = np.sqrt(res.sum())/len(d)    # L2 Norm
    tiem = end-init

    return fit, x, abs_err, tiem




if __name__=="__main__":
    #Reading in the data...
    times, lightcurve, PDC_lightcurve = read_data(sys.argv[1])

    #Fitting a N=3 polynomial...
    qr_fit, qr_x, qr_err, qr_time = LS_fitter(times, lightcurve, 3, QR=True)

    #Detrending and comparing to PDC...
    dtlc = lightcurve - qr_fit
    diff = PDC_lightcurve-dtlc-np.median(PDC_lightcurve)
    res = (diff**2).sum()**.5/len(diff)
    print()
    print("L2 Norm of detrending: ", res)
    print() ; print()

    fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True)#,sharey=True)
    ax1.plot(times,dtlc)
    ax1.set_ylabel("Polynomial detrend")
    ax1.set_ylim(dtlc.min(),dtlc.max())
    ax1.tick_params(top='off',right='off')
    cent = diff+dtlc
    ax2.plot(times, cent)
    ax2.set_ylim(cent.min(),cent.max())
    ax2.set_ylabel("PDC detrend")
    ax2.tick_params(top='off',bottom='off',right='off')

    ax3.plot(times,diff)
    ax3.set_ylim(min(diff),max(diff))
    ax3.set_ylabel("PDC-polynomial")
    ax3.set_xlabel("Times (MJD)")
    fig.subplots_adjust(hspace=0)
    fig.savefig("hw7_fig1.png")
    #ax.spines['top'].set_color('none')
    #ax.spines['bottom'].set_color('none')
    #ax.spines['left'].set_color('none')
    #ax.spines['right'].set_color('none')
    

    #Uniformly gridding everything out...
    t_rng = times.max()-times.min()
    times_1024 = np.arange(1024)*t_rng/1023.+times.min()
    dtlc_1024 = np.interp(times_1024, times, dtlc)
    
    fig, (ax1,ax2) = plt.subplots(2,sharex=True,sharey=True)
    ax1.plot(times,dtlc)
    ax1.set_ylabel("Polynomial detrend")
    ax1.tick_params(top='off',bottom='off',right='off')
    ax1.locator_params(nbins=5,axis="y")
    ax2.plot(times_1024, dtlc_1024)
    ax2.locator_params(nbins=5,axis="y")
    ax2.set_ylabel("Interpolated")
    ax2.set_xlabel("Times (MJD)")
    fig.subplots_adjust(hspace=0)
    fig.savefig("hw7_fig2.png")


    #Working with FFTs
    init = time()
    dtlc_fft = fft(dtlc_1024)
    end = time()
    dt_fft = end-init
    print("The FFT took ", dt_fft," seconds to run")
    print() ; print()


    shft =  dtlc_fft.shape[0]/2
    dtlc_ffts = np.roll(dtlc_fft,int(shft))
    #dtlc_ps = np.abs(dtlc_ffts)**2
    dtlc_ps = dtlc_ffts*dtlc_ffts.conj()
    tstep = (times_1024[1]-times_1024[0])*24*3600.
    freq_1024 = np.linspace(-1/(2*tstep), 1/(2*tstep), 1024)
    peaks = np.sort(dtlc_ps[shft:])[-2:].real
    #this wouldn't work but, there aren't that many values around the peaks...so yay for easy coding
    freq_peaks = np.array([freq_1024[np.where(dtlc_ps == i)] for i in peaks])
    t_peaks = 1./freq_peaks/3600./24.

    print("The maximum peaks are located at ", freq_peaks," uHz")
    print()
    print("These frequencies correspond to times of ", t_peaks, " days")
    
    fig, ax = plt.subplots(1)
    ax.plot(freq_1024[shft:]*1e6,dtlc_ps[shft:])
 
    for i in range(2):
        ax.annotate(str((1e6*freq_peaks[i][0]).round(2))+" uHz",
                    xy=(freq_peaks[i]*1e6, peaks[i]*(.1+.15*i)),
                    xytext=(freq_peaks[1]*1e6*2, peaks[i]*(.07+.20*i)),
                    arrowprops=dict(facecolor='black', shrink=0.05,width=2))
                   
    ax.set_xlim(0,freq_1024.max()*1e6)
    ax.set_ylim(dtlc_ps.min(),dtlc_ps.max())
    ax.set_yscale("log")
    ax.set_xlabel("Frequency ($\mu$Hz)")
    ax.set_ylabel("Power")
    fig.savefig('hw7_fig3.png')


    long_times = np.concatenate((times-t_rng, times,times+t_rng))
    long_dtlc = np.concatenate((dtlc,dtlc,dtlc))
    fig= plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.plot(np.zeros(10)+times[0],(np.arange(10)-5)*2000.,'r')
    ax.plot(np.zeros(10)+times[-1],(np.arange(10)-5)*2000.,'r')
    ax.plot(np.zeros(10)+times[0]-10,(np.arange(10)-5)*2000.,'g')
    ax.plot(np.zeros(10)+times[-1]+10,(np.arange(10)-5)*2000.,'g')
    ax.plot(long_times,long_dtlc)
    ax.set_xlim(long_times.min(),long_times.max())
    ax.set_ylim(long_dtlc.min(),long_dtlc.max())
    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("Lightcurve")
    fig.savefig("hw7_fig4.png")
