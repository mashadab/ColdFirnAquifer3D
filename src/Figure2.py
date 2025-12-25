#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:09:06 2022

@author: afzal-admin
"""

#Coding the unconfined aquifer with constant porosity in firn
#Mohammad Afzal Shadabred
#Date modified: 04/28/2022

import sys
sys.path.insert(1, '../Solver')

# import personal libraries and class
from classes import *    

from build_gridfun2D import build_grid
from build_opsfun2D_optimized import build_ops
from comp_mean_matrix import comp_mean_matrix
from solve_lbvpfun_optimized import solve_lbvp
from build_bndfun_optimized import build_bnd
from comp_fluxfun import comp_flux
from flux_upwindfun2D_optimized import flux_upwind
from Barenblatt_cartesian import Barenblatt_cartesian

##Simulation parameters
simulation_name ='vertically-integrated-model-cold-firn'
tmax = 10*365.25*day2s #10 years max #Maximum time (s)
deg2rad = np.pi/180

##Problem parameters
R = 0#0.048/day2s #Recharge (m/s)
h_top = 10; x_right = 100  #Top and right of the melted firn, to set initial condition (m)
n = 3       #Power law exponent porosity permeabity relation
Delta_rho = 998.775 #Density difference between water and gas (kg/m^3) 
k0 = 5.6e-11#absolute permeability m^2 in pore space Meyer and Hewitt 2017
mu = 1e-3   #Viscosity of water (Pa.s)
g  = 9.81  #Acceleration due to gravity (m/s^2)
phi_orig = 0.7  #Porosity of the firn/snow

############################################################
#new code (cold firn aquifer)
############################################################
T  = -30 # Temperature of the aquifer [C]
Delta_phi = 0.005790117523609654*(1-phi_orig) * (0 -T)  #calculating refrozen water (From Clark et al.,2017)
phi0 = phi_orig - Delta_phi

S  = phi0**(n-1)*Delta_rho*g*k0/mu  #Constant in constant porosity model (Huppert and Woods, 1994)
S1 = phi0**(n-1)*Delta_rho*g*k0/mu  #Constant in constant porosity model (Huppert and Woods, 1994)
S2 = phi0**(n)*Delta_rho*g*k0/(mu*(phi0+Delta_phi))  #Constant in constant porosity model (Huppert and Woods, 1994)
Nt = 200000    #Total number of time steps
dt = tmax/Nt #Length of time step (s)
kappabykappa1 = S2/S1 #the kappa ratio

print('The kappa ratio is kappa/kappa1 = ',kappabykappa1)

def S_func(dhdt):
    S = (1-np.heaviside(dhdt,0))*S1 + np.heaviside(dhdt,0)*S2
    S = spdiags(np.transpose(S), [0], Grid.N, Grid.N)
    return S


##Analytic solution initialize
g_solution, beta = Barenblatt_cartesian(kappabykappa1)
BB  = 10
kappa1 = S1/2
AA   = BB**2/kappa1
alpha = 1 - 2*beta
t_init = 1*yr2s  #initial time -- 1 day

h_init = g_solution*AA/t_init**alpha
r_init = BB*t_init**beta

print(S1)
#print(Delta_phi)
print(phi0,Delta_rho,g,k0,mu)
'''
print(kappa1)
print(AA,t_init,alpha)

print(beta, g_solution)

print(t_init, r_init, h_init)
'''

L = 2.5*r_init   #Length of the glacier (m)
############################################################

##Build grid and operator
Grid.xmin = 0; Grid.xmax = L; Grid.Nx = 250
Grid.ymin = 0; Grid.ymax = 1; Grid.Ny = 2

Grid    = build_grid(Grid)
[D,G,I] = build_ops(Grid) 
M       = comp_mean_matrix(Grid)
Xc,Yc   = np.meshgrid(Grid.xc,Grid.yc)
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

##Boundary conditions
#Fixed head at right BC
BC.dof_dir   = Grid.dof_xmax
BC.dof_f_dir = Grid.dof_f_xmax
BC.g         = np.zeros((Grid.Ny,1))

#No flow at left BC
BC.dof_neu   = Grid.dof_xmin
BC.dof_f_neu = Grid.dof_f_xmin
BC.qb        = 0*np.ones((Grid.Ny,1))

v = np.ones((Grid.Nf,1)) #Velocity for upwind
v[Grid.Nfx:] = 0 #Zero-velocity in y-direction
A = flux_upwind(v,Grid) #Same flux always advection due to slopee

##Operators
IM = I
#EX = lambda dt, h: I + dt*S*np.cos(deg2rad*0)*D@(spdiags(np.transpose(M@h), [0], Grid.Nf, Grid.Nf))@G \
#                     - dt*S*np.sin(deg2rad*0)*D@A
##########################################################################################
EX = lambda dt, h, dhdt_ind: I + dt*np.cos(deg2rad*0)*S_func(dhdt_ind)@D@(spdiags(np.transpose(M@h), [0], Grid.Nf, Grid.Nf))@G \
                               - dt*np.sin(deg2rad*0)*S_func(dhdt_ind)@D@A    
##########################################################################################                     
R  = R*np.ones((Grid.N,1))
#Kd = S*phi0*sp.eye(Grid.Nf)

##Initial condition
h = np.zeros((Grid.N,1))
#h[Xc_col<x_right] = h_top #10m high
##Analytic solution initialize
h[:100*Grid.Ny] = np.transpose(np.kron(h_init[1::100],np.ones((1,Grid.Ny))))

###########


#Storage arrays
h_sol = np.copy(h)
t     = [t_init]
time  =  t_init
h_max = [np.max(h)]
r_max = [r_init]

for i in range(0,Nt):

    # q = comp_flux(D, Kd, G, h, R, Grid, BC)

    [B,N,fn] = build_bnd(BC, Grid, I)
    #h = solve_lbvp(IM, EX(dt,h) @ h + dt*(R +fn), B, BC.g, N)
    ##########################################################################################  
    dhdt_ind = -np.ones_like(h)
    h_dummy  = solve_lbvp(IM, EX(dt,h,dhdt_ind) @ h + dt*(R +fn), B, BC.g, N)
    dhdt_ind = np.sign(h_dummy - h)
    #print('hi')
    h = solve_lbvp(IM, EX(dt,h,dhdt_ind) @ h + dt*(R +fn), B, BC.g, N)    
                         
    ##########################################################################################    
    
    time = time+dt
    
    if (i+1)%(Nt/100)==0:
        print(i+1,time/day2s, 'days')
        t = np.append(t,time)
        h_sol = np.hstack([h_sol,h])

        ##########################################################################################          
        #maximum values for analysis
        arr = h[0:Grid.N-1:2]>1e-6
        r_max = np.append(r_max,np.max(Grid.xc[arr[:,0]]))
        h_max = np.append(h_max,np.max(h))                       
        ##########################################################################################  


######################################################################
#Saving the data
######################################################################
np.savez(f'{simulation_name}_{Grid.Nx}by{Grid.Ny}_T{T}C.npz', t=t,h_sol=h_sol,r_max=r_max,h_max=h_max, Grid_xmax = Grid.xmax, Grid_Nx = Grid.Nx, Grid_Ny = Grid.Ny, Grid_xc = Grid.xc, Grid_yc = Grid.yc)

plt.figure(figsize=(10,10),dpi=100)
plt.fill_between(Grid.xc/1e3, np.transpose(h_sol[:,-1].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue, y2=0)
plt.ylabel(r'$h$ [m]')
plt.xlabel(r'$x$ [m]')
plt.tight_layout()
plt.savefig(f'../Figures/{simulation_name}_{0}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h.pdf',bbox_inches='tight', dpi = 600)


#Analytic solution
if 0 ==0:
    Q_0  =  h_top*x_right*phi0 #Volume per unit depth of water (but only half is required)
    x    =  lambda t,xi: xi*(Q_0*S*t*np.cos(0*deg2rad))**(1/3) + S*t*np.sin(0*deg2rad)
else:    
    Q_0  =  h_top*x_right*phi0/2 #Volume per unit depth of water (but only half is required)
    x    =  lambda t,xi: xi*(Q_0*S*t*np.cos(0*deg2rad))**(1/3) + S*t*np.sin(0*deg2rad) +x_right/2
xi_0 =  lambda phi_0: (9/phi_0)**(1/3)
xi_0 =  lambda phi_0: (9/phi_0)**(1/3)
f0   =  lambda xi,xi_0: (xi_0**2 - xi**2)/6  #Only for gamma = 0 
h_func    =  lambda t,xi,phi_0 : (Q_0**2/(S*np.cos(0*deg2rad)*t))**(1/3) * f0(xi,xi_0(phi_0))



'''
#New Contour plot with analytical
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(h_sol) # frame number of the animation from the saved file
tday = t/day2s
xi   = np.linspace(-xi_0(phi0),xi_0(phi0),1000)
def update_plot(frame_number, zarray, plot,t):
    plt.clf()
    plt.xlabel(r'$x$ [km]')
    plt.ylabel(r'$h$ [m]')
    plt.xlim([Grid.xmin, Grid.xmax/1e3])
    plt.ylim([0, np.max(h_sol)])
    plot[0] = plt.fill_between(Grid.xc/1e3, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue, y2=0)
    plt.plot(x(t[frame_number]*day2s,xi)/1e3,h_func(t[frame_number]*day2s,xi,phi0),'r-')
    plt.title("t= %0.2f days" % tday[frame_number],loc = 'center', fontsize=18)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig = plt.figure(figsize=(10,10) , dpi=100)
plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$h$ [m]')
plt.xlim([Grid.xmin, Grid.xmax/1e3])
plt.ylim([0, np.max(h_sol)])
plot = [plt.fill_between(Grid.xc/1e3, np.transpose(h_sol[:,0].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue, y2=0)]
plt.title("t= %0.2f days" % tday[0],loc = 'center', fontsize=18)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(h_sol[:,:], plot[:],tday[:]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{0}degree__tf{t[frn-1]}.mov", writer='ffmpeg', fps=30)
'''


'''
#New Contour plot
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(h_sol) # frame number of the animation from the saved file
tday = t/day2s
def update_plot(frame_number, zarray, plot,t):
    plt.clf()
    plt.xlabel(r'$x$ [km]')
    plt.ylabel(r'$h$ [m]')
    plt.xlim([Grid.xmin, Grid.xmax/1e3])
    plt.ylim([0, np.max(h_sol)])
    plot[0] = plt.fill_between(Grid.xc/1e3, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue, y2=0)
    plt.title("t= %0.2f days" % tday[frame_number],loc = 'center', fontsize=18)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig = plt.figure(figsize=(10,10) , dpi=100)
plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$h$ [m]')
plt.xlim([Grid.xmin, Grid.xmax/1e3])
plt.ylim([0, np.max(h_sol)])
plot = [plt.fill_between(Grid.xc/1e3, np.transpose(h_sol[:,0].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue, y2=0)]
plt.title("t= %0.2f days" % tday[0],loc = 'center', fontsize=18)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(h_sol[:,:], plot[:],tday[:]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{0}degree__tf{t[frn-1]}T_{T}C.mov", writer='ffmpeg', fps=30)
'''




#plt.plot(t/yr2s,r_max,'ro')

indices = 70
tlog = np.log(t[1:])
th_dimless = np.log(h_max[1:]/h_max[0])
tr_dimless = np.log(r_max[1:]/r_max[0])
# Perform linear fit
coefficients = np.polyfit(tlog[indices:], tr_dimless[indices:], 1)
m_r = coefficients[0]
b_r = coefficients[1]

print("Slope (m):", m_r)
print("Y-intercept (b):", b_r)

# Perform linear fit
coefficients = np.polyfit(tlog[indices:], th_dimless[indices:], 1)
m_h = coefficients[0]
b_h = coefficients[1]

print("Slope (m):", m_h)
print("Y-intercept (b):", b_h)


th_dimless = np.log10(h_max/h_max[0])
tr_dimless = np.log10(r_max/r_max[0])

plt.figure()
plt.plot(np.log(t),np.log(r_max/r_max[0]),'r.')



red    = [190/255 ,30/255 ,45/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]

import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})
from scipy.sparse import spdiags

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
plt.subplot(1,3,1)
#plt.loglog(t,h_max_analy,'r-',linewidth=3)
plt.plot(tlog,th_dimless[:-1],'b-',linewidth=5,label='Num',color=blue,mfc='none',markersize=10)
plt.plot(tlog, np.log10((t_init/t[:-1])**alpha),'k--',linewidth=3,label=r'Ana',alpha=0.75)
plt.ylabel(r'$log[h_{max}/h_{max}(t=0)]$')
plt.xlabel(r'log t [log days]')
#plt.ylim([-0.6,0.05])
plt.legend(loc='best',framealpha=0.0)
#plt.axis('scaled')

plt.subplot(1,3,2)
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(tlog,tr_dimless[:-1],'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(tlog, np.log10((t[:-1]/t_init)**beta),'k--',linewidth=3,label=r'Analytic',alpha=0.75)
plt.xlabel(r'$t$')
plt.ylabel(r'$log[r_{max}/r_{max}(t=0)]$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r'log t [log days]')
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)


plt.subplot(1,3,3)

Vol = np.sum(h_sol,0)*Grid.dx*phi0
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(tlog,np.log10(Vol[:-1]/Vol[0]),'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(tlog, np.log10((t[:-1]/t_init)**(3*beta-1)),'k--',linewidth=3,label=r'Analytic',alpha=0.75)
plt.xlabel(r'$t$')
plt.ylabel(r'$log[Q_{max}/Q_{max}(t=0)]$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r'log t [log days]')
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)
plt.tight_layout()
#plt.xlim([Grid.xmin, 10])
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.savefig(f'../Figures/max_length_height_mound_cold_T{T}C.pdf',bbox_inches='tight', dpi = 600)



#Non-dimensional
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
plt.subplot(1,3,1)
#plt.loglog(t,h_max_analy,'r-',linewidth=3)
plt.plot(np.log10(t[1:]/t[0]),th_dimless[:-1],'b-',linewidth=5,label='Num',color=blue,mfc='none',markersize=10)
plt.plot(np.log10(t[1:]/t[0]), np.log10((t_init/t[:-1])**alpha),'k--',linewidth=3,label=r'Ana',alpha=0.75)
plt.ylabel(r'log $( h_{max}/h_{max,0})$')
plt.xlabel(r'log (t/t$_0)$')
#plt.ylim([-0.6,0.05])
plt.legend(loc='best',framealpha=0.0)
#plt.axis('scaled')

plt.subplot(1,3,2)
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(np.log10(t[1:]/t[0]),tr_dimless[:-1],'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(np.log10(t[1:]/t[0]), np.log10((t[:-1]/t_init)**beta),'k--',linewidth=3,label=r'Analytic',alpha=0.75)
plt.xlabel(r'$t$')
plt.ylabel(r'log $(r_{max}/r_{max,0})$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r'log (t/t$_0$)')
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)


plt.subplot(1,3,3)

Vol = np.sum(h_sol,0)*Grid.dx*phi0
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(np.log10(t[1:]/t[0]),np.log10(Vol[:-1]/Vol[0]),'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(np.log10(t[1:]/t[0]), np.log10((t[:-1]/t_init)**(3*beta-1)),'k--',linewidth=3,label=r'Analytic',alpha=0.75)
plt.xlabel(r'$t$')
plt.ylabel(r'log $(Q/Q_{0})$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r'log (t/t$_0)$')
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)
plt.tight_layout()
#plt.xlim([Grid.xmin, 10])
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.savefig(f'../Figures/max_length_height_mound_cold_T{T}C_nonDim.pdf',bbox_inches='tight', dpi = 600)







plt.figure(figsize=(12,4),dpi=100)
Num = 6
for i in np.linspace(0,np.shape(h_sol)[1]-1,Num).round().astype(int):
    if i >0:
        plt.plot(Grid.xc/1e3, np.transpose(h_sol[:,i].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue,linewidth=5,alpha=((i+10)/(np.shape(h_sol)[1]-1+20)),label=r'%0.1f yrs'%(t[i]/yr2s))
        #analytic solution
        h_analy = h_init*(t_init/t[i])**alpha
        x_analy = np.linspace(0,r_init,len(h_init))*(t[i]/t_init)**beta
        plt.plot(x_analy/1e3, h_analy,'k--',linewidth=5,alpha=((i+10)/(np.shape(h_sol)[1]-1+20)))
plt.ylabel(r'$h$ [m]')
plt.xlabel(r'$x$ [km]')
plt.tight_layout()
plt.legend(loc='best',ncol=3)
plt.savefig(f'../Figures/{simulation_name}_{0}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h_combined_T{T}C.pdf',bbox_inches='tight', dpi = 600)

