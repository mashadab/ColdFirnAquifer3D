#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 09:24:14 2025

@author: ms6985
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})

#Colors
brown  = [181/255 , 101/255, 29/255]
red    = [190/255 ,30/255 ,45/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
brown  = [155/255 ,  118/255 ,  83/255]
tan    = [199/255 , 178/255 , 153/255]
gray   = [100/255 , 100/255 , 100/255]

kappabykappa1_arr = np.linspace(0.2,1,49)#np.array([0.321,0.9502857142857141])
beta_arr = np.array([])
for kappabykappa1 in kappabykappa1_arr:
    # --- Constants ---
    N = 10000        # number of grid points
    h = 1.0 / N    # step size
    zeta = np.linspace(0, 1, N + 1)
    
    kappa1bykappa = 1/kappabykappa1
    
    # --- Function to compute residuals for a given beta ---
    def solve_g(beta):
        g = ((1-zeta**2)).copy()#np.ones(N + 1)
        #Boundary conditions
        g[N] = 0
        g[N-1] = g[N]+kappa1bykappa*beta*h/2 #Nuemann condition at the last cell
    
        for kkk in range(0,2):
            # Solve backward from i = N-2 to i = 0
            for i in reversed(range(N - 1)):
                z = zeta[i + 1]
        
                A = 2/(beta*z*h)
                B = 2*(1-2*beta)*h/(beta*z)
            
                #g[i] =  A/kappa1bykappa*(g[i + 2]**2 - 2 * g[i + 1]**2 + g[i]**2) + B * g[i+1] + g[i+2]  # rough fix #weird line
                if g[i + 2]**2 - 2 * g[i + 1]**2 + g[i]**2 > 0 :
                    kappa1bykappa_new =  kappa1bykappa
                    #print('Y - kappa ratio used')
                else: 
                    kappa1bykappa_new =  1.0       
                    #print('N - kappa ratio not used')
                #solve using quadratic forbetala
                AA = A/kappa1bykappa_new
                BB = -1
                CC = A/kappa1bykappa_new*(g[i + 2]**2 - 2 * g[i + 1]**2 ) + B * g[i+1] + g[i+2]
                g[i] = (-BB + np.abs(np.sqrt(BB**2 - 4*AA*CC)))/(2*AA)
    
        return g
    
    # --- Objective function for shooting: dg/dzeta ≈ 0 at zeta = 0 ---
    def shooting_residual(beta):
        g = solve_g(beta)
        return (g[1] - g[0]) / h 
    
    
    # --- Root finding for beta ---
    beta_guess_low  = 0.15+(0.3 - 0.15)/(1-0.1) * (kappabykappa1-0.15)
    beta_guess_high = 0.5
    
    result = root_scalar(shooting_residual, bracket=[beta_guess_low, beta_guess_high], method='brentq')
    
    if result.converged:
        beta_star = result.root
        print(f"k/k1={kappabykappa1}, Eigenvalue 	β = {beta_star:.6f}")
        
        g_solution = solve_g(beta_star)
        
        # Optional: Recover f = g^{2/3}
        #f_solution = g_solution**(2/3)
        
        # --- Plotting ---
        plt.figure(figsize=(8, 4))
        #plt.plot(zeta, f_solution, label='f(ζ)', color='dodgerblue')
        if kappabykappa1==1.0:
            plt.plot(zeta, g_solution, '-', color=red,label="Numeric")
            plt.plot(zeta,(1-zeta**2)/12,'--',color=blue,label="Analytic")
            plt.legend()
        else:
            plt.plot(zeta, g_solution, '-', color=red)
        plt.xlabel('ζ')
        plt.ylabel('Solution, Φ_1')
        plt.title(f"Solution for β ≈ {beta_star:.4f}")
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
    else:
        print("Failed to converge to a solution for beta.")
    
    beta_arr = np.append(beta_arr,result.root) 
  


'''
beta_vals = np.linspace(beta_guess_low, beta_guess_high, 100)
res_vals = [shooting_residual(beta) for beta in beta_vals]

plt.figure()
plt.plot(beta_vals, res_vals)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('μ')
plt.ylabel("Residual at ζ=0 (g[1]-g[0])/h")
plt.title('Shooting Residual vs μ')
plt.grid(True)
plt.show()
'''

plt.figure()
plt.plot(kappabykappa1_arr,beta_arr,color=red)
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\kappa$ / $\kappa_1$')
plt.title('Cartesian coordinates')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f'beta_vskapparatio_Cart.pdf',bbox_inches='tight', dpi = 600)



#####################################################################################
#Analysis points
#####################################################################################
beta_arr_new = np.array([])
T=[0,-10,-30,-50,-100]
koverk1=[1,0.9751428571428573,0.9254285714285713,0.8757142857142858,0.7514285714285711]

for kappabykappa1 in koverk1:
    # --- Constants ---
    N = 10000        # number of grid points
    h = 1.0 / N    # step size
    zeta = np.linspace(0, 1, N + 1)
    
    kappa1bykappa = 1/kappabykappa1
    
    # --- Function to compute residuals for a given beta ---
    def solve_g(beta):
        g = ((1-zeta**2)).copy()#np.ones(N + 1)
        #Boundary conditions
        g[N] = 0
        g[N-1] = g[N]+kappa1bykappa*beta*h/2 #Nuemann condition at the last cell
    
        for kkk in range(0,2):
            # Solve backward from i = N-2 to i = 0
            for i in reversed(range(N - 1)):
                z = zeta[i + 1]
        
                A = 2/(beta*z*h)
                B = 2*(1-2*beta)*h/(beta*z)
            
                #g[i] =  A/kappa1bykappa*(g[i + 2]**2 - 2 * g[i + 1]**2 + g[i]**2) + B * g[i+1] + g[i+2]  # rough fix #weird line
                if g[i + 2]**2 - 2 * g[i + 1]**2 + g[i]**2 > 0 :
                    kappa1bykappa_new =  kappa1bykappa
                    #print('Y - kappa ratio used')
                else: 
                    kappa1bykappa_new =  1.0       
                    #print('N - kappa ratio not used')
                #solve using quadratic forbetala
                AA = A/kappa1bykappa_new
                BB = -1
                CC = A/kappa1bykappa_new*(g[i + 2]**2 - 2 * g[i + 1]**2 ) + B * g[i+1] + g[i+2]
                g[i] = (-BB + np.abs(np.sqrt(BB**2 - 4*AA*CC)))/(2*AA)
    
        return g
    
    # --- Objective function for shooting: dg/dzeta ≈ 0 at zeta = 0 ---
    def shooting_residual(beta):
        g = solve_g(beta)
        return (g[1] - g[0]) / h 
    
    
    # --- Root finding for beta ---
    beta_guess_low  = 0.15+(0.3 - 0.15)/(1-0.1) * (kappabykappa1-0.15)
    beta_guess_high = 0.5
    
    result = root_scalar(shooting_residual, bracket=[beta_guess_low, beta_guess_high], method='brentq')
    
    if result.converged:
        beta_star = result.root
        print(f"k/k1={kappabykappa1}, Eigenvalue 	β = {beta_star:.6f}")
        
        g_solution = solve_g(beta_star)
        
        # Optional: Recover f = g^{2/3}
        #f_solution = g_solution**(2/3)
        
        # --- Plotting ---
        plt.figure(figsize=(8, 4))
        #plt.plot(zeta, f_solution, label='f(ζ)', color='dodgerblue')
        if kappabykappa1==1.0:
            plt.plot(zeta, g_solution, '-', color=red,label="Numeric")
            plt.plot(zeta,(1-zeta**2)/12,'--',color=blue,label="Analytic")
            plt.legend()
        else:
            plt.plot(zeta, g_solution, '-', color=red)
        plt.xlabel('ζ')
        plt.ylabel('Solution, Φ_1')
        plt.title(f"Solution for β ≈ {beta_star:.4f}")
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
    else:
        print("Failed to converge to a solution for beta.")
    
    beta_arr_new = np.append(beta_arr_new,result.root) 


plt.figure(figsize=(5,4))
plt.plot(kappabykappa1_arr,beta_arr,'k--',linewidth=2.5)
plt.plot(koverk1[0],beta_arr_new[0],'ro',color=red,markersize=15)
plt.plot(koverk1[1],beta_arr_new[1],'ro',color=brown,markersize=15)
plt.plot(koverk1[2],beta_arr_new[2],'ro',color=blue,markersize=15)
plt.plot(koverk1[3],beta_arr_new[3],'ro',color=purple,markersize=15)
plt.plot(koverk1[4],beta_arr_new[4],'ro',color=green,markersize=15)
plt.xlim([0.6,1.02])
plt.ylim([0.3,0.335])
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\kappa$ / $\kappa_1$')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(f'beta_vskapparatio_Cart.pdf',bbox_inches='tight', dpi = 600)