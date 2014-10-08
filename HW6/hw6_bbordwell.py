#!/usr/bin/python3.4


######Importing##Useful##Packages###################################
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from copy import deepcopy
import time
import pyfits
from os import sys
import pdb
from time import time as time



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



######A##function##to##iteratively##solve##the##normal##equations######
def iter_fitter(x_data, y_data, N, 
                GS=True, J=False, SOR=False, CG=False, 
                tol=1e-8, w=1.9):
    """
    Purpose: Performs a least squares polynomial fit of degree N to the data
    
    Input: 
    x_data, y_data = Independent and dependent data values 
    N = Degree of polynomial being fitted
    /GS = A keyword option to use the Gauss-Seidel approach to finding the M matrix.
    /J = A keyword option to use the Jacobi approach to finding the M matrix.
    /SOR = A keyword option to use the SOR approach to finding the M matrix.
    /CG = A keyword option to use the conjugate gradient approach.
    tol = Tolerance for the iteration to converge.
    w = Omega constant for SOR.

    Output:
    fit = The fitted data.
    x = The coefficients of the polynomial fit.
    abs_err = L2 norm for the fit.
    steps = The number of steps it took to reach convergence.
    time = The length of time the solver took to converge.
    rks = The residuals produced for each iterative step.
    """

    #Preconditioning and forming the matrix...
    x_data = (x_data-x_data.min())/(x_data.max()-x_data.min())
    A = np.array([x_data**i for i in range(N+1)])
    ata = np.dot(A,A.transpose())
    b = np.dot(y_data,A.transpose())


    method = {0: lambda x: np.tril(x),                             # Gauss-Seidel
              1: lambda x: np.diag(np.diag(x)),                    # Jacobi
              2: lambda x: (1-w)*np.diag(np.diag(x))+w*np.tril(x)} # SOR


    #Establishing the matrix to be used during the iterations...
    Mi = np.linalg.inv(method[np.where([GS,J,SOR])[0][-1]](ata))
    if SOR: Mi *= w


    def residual(xk,A=ata,b=b):
        return b-np.dot(xk,A)


    #Setting up initial values for the iteration
    xk = np.append(np.mean(b),np.zeros(N))  
    diff = np.dot(residual(xk),Mi)
    steps = 0
    rks = np.array([0,0,0])
    rks = np.append([rks],[residual(xk)],0)


    #Mirroring the book for the conjugate gradient method
    if CG:
        def conj_grad(rk, iM=Mi, A=ata):
            pk = np.dot(rk,iM)
            ak = np.dot(rk,rk.transpose())/np.dot(pk,np.dot(pk,A))
            return pk*ak

        chk = tol**2*np.dot(b,b.transpose())
        init = time()
        while np.dot(diff,diff.transpose()) > chk: 
            xk += diff
            rks = np.append(rks,[residual(xk)],0)
            diff = conj_grad(rks[-1])

            steps += 1

    #Otherwise keeping it nice and concise
    else: 
        chk = tol*np.linalg.norm(b,2)
        init = time()

        while np.linalg.norm(rks[-1],2) > chk: 
            xk += np.dot(rks[-1],Mi)
            rks = np.append(rks,[residual(xk)],0)

            steps += 1

    end = time()  
    rks = rks[1:]
    x = xk


    #Finding the fit and residuals...
    fit = np.dot(x.transpose(),A)  ;  res = (y_data-fit)**2
    if len(fit.shape) > 1: d = fit[0]
    else: d = fit

    abs_err = np.sqrt(res.sum())/len(d)    # L2 Norm
    tiem = end-init
    
    return fit, x, abs_err, steps, tiem, rks 



#####Defining##a##criterion##to##check##for##SigFig##agreement############## 
def crit(num, digits=3):
    if np.isscalar(num):
        if num == 0: return '0'
        else:        
            num = np.abs(num)
            num /= 10**np.floor(np.log10(num))
            return str(num)[0:digits+1]
    else:
        sf = np.array([])
        for a in num:
            if a == 0: sf = np.append(sf,'0')
            else: 
                a = np.abs(a)
                a /= 10**np.floor(np.log10(a))
                sf = np.append(sf,str(a)[0:digits+1])
        return sf



#####Setting##up##the##code##to##run##all##the##specific##stuff###########
if __name__=="__main__":
    #Reading in the data and taking the system argument for the polynomial degree to fit...
    times, lightcurve, PDC_lightcurve = read_data(sys.argv[1])
    N = int(sys.argv[2])

    #Plotting the normal equations and qr decomposition solutions to the LS problem...
    #fig = plt.figure()
    #ax = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)
    timed = []  ;  err = []
    for i in range(1,N+1):
        ls_fit, ls_x, ls_err, ls_time = LS_fitter(times, lightcurve, i)
        qr_fit, qr_x, qr_err, qr_time = LS_fitter(times, lightcurve, i, QR=True)

        #Saving these for later comparison...
        if i == 2: 
            N2_sol = qr_x
            N2_sol_ls = ls_x

        #Getting some matrices for other plots...
        timed.append([ls_time,qr_time])
        err.append([ls_err,qr_err])
    #    ax.plot(times, qr_fit-ls_fit, label="N = "+str(i))


    #Making plots for the first question...
    err = np.array(err).transpose()
    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("QR fit - normal equations approach (e-/sec)",fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("Difference between fitting methods with polynomial degree N",fontsize=8)
    ax.legend(loc='upper right', ncol = int(N/4),fontsize=8)
    ax2.plot(np.arange(N)+1, err[1]-err[0])
    ax2.tick_params(labelsize=8)
    ax2.set_xlabel("Polynomial degree N",fontsize=8)
    ax2.set_ylabel("QR fit error- normal equations error (e-/sec)",fontsize=8)
    ax2.set_title("Difference between fitting methods with polynomial degree N", fontsize=8)
    fig.tight_layout()
    fig.savefig('hw5_fig1.png')
    fig.clf()
    

    #Plotting the length of time the solution required for different values of N
    timed = np.array(timed).transpose()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(N)+1, timed[0], 'r--', label="QR Decomposition Approach")
    ax.plot(np.arange(N)+1, timed[1], 'b--', label="Normal Equations Approach")
    ax.set_xlabel("Polynomial degree N")
    ax.set_ylabel("Duration of fit (s)")
    ax.set_title("Runtime variations")
    ax.legend(loc='upper left')
    fig.savefig('hw5_fig2.png')
    


    #Attempting iterative solves...
    #For SOR...
    tolr = 1e-2 # setting a starting tolerance...
    print("Attempting tolerance: "+str(tolr))

    sor_x = np.zeros(len(N2_sol))
    while (crit(sor_x) == crit(N2_sol)).sum() != len(N2_sol): #sigfig criterion, default 3
        sor_fit, sor_x, sor_err, sor_n, sor_t, sor_conv = iter_fitter(
            times, lightcurve, 2, SOR=True, tol=tolr)

        print("True solution: \n\t", N2_sol)
        print("Iterative solution: \n\t", sor_x)
        print("In ", sor_n," steps and ", sor_t, " seconds")

        tolr /= 10        
        print("Attempting tolerance: "+str(tolr))
        
    tolr *= 10        
    print("Necessary tolerance: "+str(tolr))


    #For Gauss-Seidel (was once conjugate gradient, hence the cg notation...
    tolr = 1e-2  # setting a starting tolerance...
    print("Attempting tolerance: "+str(tolr))

    cg_x = np.zeros(len(N2_sol))
    while (crit(cg_x) == crit(N2_sol)).sum() != len(N2_sol): 
        cg_fit, cg_x, cg_err, cg_n, cg_t, cg_conv = iter_fitter(
            times, lightcurve, 2, GS=True, tol=tolr)

        print("True solution: \n \t "+str(N2_sol))
        print("Iterative solution: \n \t"+ str(cg_x))
        print("In ", cg_n," steps and ", cg_t, " seconds")

        tolr /= 10
        print("Attempting tolerance: "+str(tolr))

    tolr *= 10        
    print("Necessary tolerance: "+str(tolr))


    #Reporting back for safety...
    print("Previous solutions were, ")
    print("\t QR: "+str(N2_sol))
    print("\t NE: "+str(N2_sol_ls))

    print("The SOR fit found: ")
    print("\t x = "+str(sor_x))
    print("\t in "+str(sor_n)+" steps")
    print("\t in "+str(sor_t)+" seconds")

    print("The Gauss-Seidel found: ")
    print("\t x = "+str(cg_x))
    print("\t in "+str(cg_n)+" steps")
    print("\t in "+str(cg_t)+" seconds")

    
    #Making final convergence plot...
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(sor_n), sor_conv, label="SOR")
    ax.plot(np.arange(cg_n), cg_conv, label="Gauss-Seidel")
    ax.legend(loc = 'upper right', n_cols = 2)
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Residual")
    ax.set_title("Convergence")
    fig.savefig('hw5_fig3.png')
    pdb.set_trace()
