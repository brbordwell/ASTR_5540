#!/usr/bin/python3.4

import numpy as np
from matplotlib import pyplot as plt

def first_order_finite_diff(f, x, delta_x):
    return (f(x+delta_x)-f(x))/delta_x

def second_order_finite_diff(f, x, delta_x):
    return (f(x+delta_x)-f(x-delta_x))/(2*delta_x)


def plot_FD_error(f, dfdx, x0, ax, min_delta_x=-10, max_delta_x=-1, ddfdxx=None, dddfdxxx=None) :
    """Function will find and plot the error in the FD approximation """

    #Finding the error in the FD approximation
    delta_x = np.logspace(min_delta_x, max_delta_x, base = 10.0)
    
    FD_1_error = np.abs(first_order_finite_diff(f, x0, delta_x)-dfdx(x0))
    #print(FD_1_error) Default code

    FD_2_error = np.abs(second_order_finite_diff(f, x0, delta_x)-dfdx(x0))
        

    #Setting up the plot
    #ax.loglog(delta_x, FD_1_error, label='FD-1') Default code
    if ddfdxx != None:
        d_err = delta_x/2.*np.abs(ddfdxx(x0))
        ax.loglog(delta_x, d_err, 'r-o', label='FD-1 d_err')
    if dddfdxxx != None:
        d_err2 = delta_x**2/6.*np.abs(dddfdxxx(x0))
        ax.loglog(delta_x, d_err2, 'g-o', label='FD-2 d_err')
    ax.loglog(delta_x, FD_1_error, 'b-o', label='FD-1')
    ax.loglog(delta_x, FD_2_error, 'k-o', label='FD-2')
    ax.set_xlabel(r'$\Delta x$')
    ax.set_ylabel('Absolute error in FD approximation')
    ax.legend(loc="lower right")



if __name__=="__main__":
    
    figure_1 = plt.figure()
    ax = figure_1.add_subplot(1,1,1)
  
    #Defining the particulars of the function to be approximated
    f = np.sin
    dfdx = np.cos  
    ddfdxx = lambda x: -1*np.sin(x)
    dddfdxxx = lambda x: -1*np.cos(x)
    x0 = .25

    #plot_FD_error(f, dfdx, x0, ax) Default code
    #plot_FD_error(f, dfdx, x0, ax, -20,1) Problem 2
    #plot_FD_error(f, dfdx, x0, ax, -20,1, ddfdxx) Problem 3
    #plot_FD_error(f, dfdx, x0, ax, -20,1, ddfdxx, dddfdxxx) Problem 4
    plot_FD_error(f, dfdx, x0, ax, -15.,0., ddfdxx, dddfdxxx)
    figure_1.savefig("hw1_fig1.png")
    plt.show()
