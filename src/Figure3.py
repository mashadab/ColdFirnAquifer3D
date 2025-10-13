#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:09:06 2022

@author: afzal-admin
"""

#Coding the unconfined aquifer with constant porosity in firn
#Mohammad Afzal Shadab
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

##Simulation parameters
simulation_name ='vertically-integrated-model-cold-firn-old'
L = 1000   #Length of the glacier (m)
tmax = 10*365.25*day2s #10 years max #Maximum time (s)
deg2rad = np.pi/180

##Problem parameters
R = 0#0.048/day2s #Recharge (m/s)
tilt_angle = 0   #angle of the slope (degrees)
h_top = 10; x_right = 100  #Top and right of the melted firn, to set initial condition (m)
n = 3       #Power law exponent porosity permeabity relation
Delta_rho = 1e3 #Density difference between water and gas (kg/m^3) 
k0 = 5.6e-11#absolute permeability m^2 in pore space Meyer and Hewitt 2017
mu = 1e-3   #Viscosity of water (Pa.s)
g  = 9.808  #Acceleration due to gravity (m/s^2)
phi_orig = 0.7  #Porosity of the firn/snow

############################################################
#new code (cold firn aquifer)
############################################################
T  = -50 # Temperature of the aquifer [C]
Delta_phi = 0.0058*(1-phi_orig) * (0 -T)  #calculating refrozen water (From Clark et al.,2017)
phi0 = phi_orig - Delta_phi

S  = phi0**(n-1)*Delta_rho*g*k0/mu  #Constant in constant porosity model (Huppert and Woods, 1994)
S1 = phi0**(n-1)*Delta_rho*g*k0/mu  #Constant in constant porosity model (Huppert and Woods, 1994)
S2 = phi0**(n)*Delta_rho*g*k0/(mu*(phi0+Delta_phi))  #Constant in constant porosity model (Huppert and Woods, 1994)
Nt = 50000    #Total number of time steps
dt = tmax/Nt #Length of time step (s)
kappabykappa1 = S2/S1 #the kappa ratio

print('The kappa ratio is kappa/kappa1 = ',kappabykappa1)

def S_func(dhdt):
    S = (1-np.heaviside(dhdt,0))*S1 + np.heaviside(dhdt,0)*S2
    S = spdiags(np.transpose(S), [0], Grid.N, Grid.N)
    return S

############################################################

##Build grid and operator
Grid.xmin = 0; Grid.xmax = L; Grid.Nx = 100
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
#EX = lambda dt, h: I + dt*S*np.cos(deg2rad*tilt_angle)*D@(spdiags(np.transpose(M@h), [0], Grid.Nf, Grid.Nf))@G \
#                     - dt*S*np.sin(deg2rad*tilt_angle)*D@A
##########################################################################################
EX = lambda dt, h, dhdt_ind: I + dt*np.cos(deg2rad*tilt_angle)*S_func(dhdt_ind)@D@(spdiags(np.transpose(M@h), [0], Grid.Nf, Grid.Nf))@G \
                               - dt*np.sin(deg2rad*tilt_angle)*S_func(dhdt_ind)@D@A    
##########################################################################################                     
R  = R*np.ones((Grid.N,1))
#Kd = S*phi0*sp.eye(Grid.Nf)

##Initial condition
h = np.zeros((Grid.N,1))
h[Xc_col<x_right] = h_top #10m high

#Storage arrays
h_sol = np.copy(h)
t     = [0]
time  =  0
h_max = [np.max(h)]
r_max = [x_right]

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
np.savez(f'{simulation_name}_{Grid.Nx}by{Grid.Ny}_T{T}C.npz', t=t,h_sol=h_sol,r_max=r_max,h_max=h_max)

plt.figure(figsize=(10,10),dpi=100)
plt.fill_between(Grid.xc/1e3, np.transpose(h_sol[:,-1].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue, y2=0)
plt.ylabel(r'$h$ [m]')
plt.xlabel(r'$x$ [m]')
plt.tight_layout()
plt.savefig(f'../Figures/{simulation_name}_{tilt_angle}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h.pdf',bbox_inches='tight', dpi = 600)


#Analytic solution
if tilt_angle ==0:
    Q_0  =  h_top*x_right*phi0 #Volume per unit depth of water (but only half is required)
    x    =  lambda t,xi: xi*(Q_0*S*t*np.cos(tilt_angle*deg2rad))**(1/3) + S*t*np.sin(tilt_angle*deg2rad)
else:    
    Q_0  =  h_top*x_right*phi0/2 #Volume per unit depth of water (but only half is required)
    x    =  lambda t,xi: xi*(Q_0*S*t*np.cos(tilt_angle*deg2rad))**(1/3) + S*t*np.sin(tilt_angle*deg2rad) +x_right/2
xi_0 =  lambda phi_0: (9/phi_0)**(1/3)
xi_0 =  lambda phi_0: (9/phi_0)**(1/3)
f0   =  lambda xi,xi_0: (xi_0**2 - xi**2)/6  #Only for gamma = 0 
h_func    =  lambda t,xi,phi_0 : (Q_0**2/(S*np.cos(tilt_angle*deg2rad)*t))**(1/3) * f0(xi,xi_0(phi_0))



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

ani.save(f"../Figures/{simulation_name}_{tilt_angle}degree__tf{t[frn-1]}.mov", writer='ffmpeg', fps=30)
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

ani.save(f"../Figures/{simulation_name}_{tilt_angle}degree__tf{t[frn-1]}T_{T}C.mov", writer='ffmpeg', fps=30)
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

plt.figure()
plt.plot(np.log(t),np.log(r_max/r_max[0]),'r.')



red    = [190/255 ,30/255 ,45/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,8))
plt.subplot(2,1,1)
#plt.loglog(t,h_max_analy,'r-',linewidth=3)
plt.plot(tlog,th_dimless,'bo',linewidth=10,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(tlog, tlog*m_h + b_h,'r-',linewidth=3,label=r'Fit, $\alpha=$%0.3f'%(-m_h),color=red,alpha=0.75)
plt.ylabel(r'$log[h_{max}/h_{max}(t=0)]$')
#plt.ylim([-0.6,0.05])
plt.legend(loc='best',framealpha=0.0)
#plt.axis('scaled')

plt.subplot(2,1,2)
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(tlog,tr_dimless,'bo',linewidth=10,label='Simulation',color=blue,mfc='none',markersize=10)
plt.plot(tlog, tlog*m_r + b_r,'r-',linewidth=3,label=r'Fit, $\beta=$%0.3f'%(m_r),color=red,alpha=0.75)
plt.xlabel(r'$t$')
plt.ylabel(r'$log[r_{max}/r_{max}(t=0)]$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r'log t [log days]')
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
plt.tight_layout()
plt.legend(loc='best',framealpha=0.0)
#plt.xlim([Grid.xmin, 10])
plt.savefig(f'../Figures/max_length_height_mound_cold_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
plt.subplot(2,2,1)
#plt.loglog(t,h_max_analy,'r-',linewidth=3)
plt.plot(t,h_max,'b-',linewidth=5,label='$T=%0.0f ^oC$'%(T),color=blue,mfc='none',markersize=10)
plt.ylabel(r'$h_{max} [m]$')
#plt.ylim([-0.6,0.05])
plt.legend(loc='best',framealpha=0.0)
#plt.axis('scaled')

plt.subplot(2,2,2)
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(t,r_max,'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.xlabel(r'$t [s]$')
plt.ylabel(r'$r_{max}$ [m]')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r't [s]')
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)


plt.subplot(2,2,3)
Vol = np.sum(h_sol,0)*Grid.dx*phi0
#plt.loglog(t,x_max_analy,'r-',linewidth=3)
plt.plot(t,Vol,'b-',linewidth=5,label='Simulation',color=blue,mfc='none',markersize=10)
plt.xlabel(r'$t [s]$')
plt.ylabel(r'$Q_{max} [m^3]$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r't [s]')
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)
#plt.xlim([Grid.xmin, 10])
plt.subplots_adjust(wspace=0.0, hspace=0)
plt.tight_layout()
plt.savefig(f'../Figures/max_length_height_mound_cold_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)


######################################################################
#Saving the data
######################################################################
np.savez(f'{simulation_name}_{Grid.Nx}by{Grid.Ny}_T{T}C.npz', t=t,h_sol=h_sol,r_max=r_max,h_max=h_max,dx = Grid.dx,Vol=Vol)



plt.figure(figsize=(10,4),dpi=100)
Num = 6
for i in np.linspace(0,np.shape(h_sol)[1]-1,Num).round().astype(int):
    if i >0:
        plt.plot(Grid.xc/1e3, np.transpose(h_sol[:,i].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue,linewidth=5,alpha=((i+10)/(np.shape(h_sol)[1]-1+20)),label=r'%0.1f yrs'%(t[i]/yr2s))
plt.ylabel(r'$h$ [m]')
plt.xlabel(r'$x$ [m]')
plt.tight_layout()
plt.legend(loc='best',ncol=3)
plt.savefig(f'../Figures/{simulation_name}_{tilt_angle}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h_combined_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)



plt.figure(figsize=(10,4),dpi=100)
t_arr = (np.array([0, 10, 50, 100])).astype(int)
#for i in np.linspace(0,np.shape(h_sol)[1]-1,Num).round().astype(int):
for i in t_arr:
    ii = np.argwhere(i==t_arr)[0][0]+1
    #if i >0:
    Ratio = 1-((ii+1)/(len(t_arr)+1))
    Resulting_Color = np.multiply((1 - Ratio),blue) + np.array([1,1,1]) * Ratio
    plt.plot(Grid.xc/1e3, np.transpose(h_sol[:,i].reshape(Grid.Nx,Grid.Ny))[0,:],color=Resulting_Color,linewidth=5,label=r'%0.0f yrs'%(t[i]/yr2s))
plt.ylabel(r'$h$ [m]')
plt.xlabel(r'$x$ [m]')
plt.tight_layout()
plt.legend(loc='best',ncol=2)
plt.savefig(f'../Figures/{simulation_name}_{tilt_angle}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h_combined_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)







################################################################
#Analysis script
################################################################

tyear = t/yr2s
data = np.load('vertically-integrated-model-cold-firn-old_100by2_T0C.npz')
t_0=data['t'] ;h_sol_0 =data['h_sol']; r_max_0 =data['r_max']; h_max_0 =data['h_max']; Vol_0 =data['Vol']

data = np.load('vertically-integrated-model-cold-firn-old_100by2_T-10C.npz')
t_10=data['t'] ;h_sol_10 =data['h_sol']; r_max_10 =data['r_max']; h_max_10 =data['h_max']; Vol_10 =data['Vol']

data = np.load('vertically-integrated-model-cold-firn-old_100by2_T-20C.npz')
t_20=data['t'] ;h_sol_20 =data['h_sol']; r_max_20 =data['r_max']; h_max_20 =data['h_max']; Vol_20 =data['Vol']

data = np.load('vertically-integrated-model-cold-firn-old_100by2_T-50C.npz')
t_50=data['t'] ;h_sol_50 =data['h_sol']; r_max_50 =data['r_max']; h_max_50 =data['h_max']; Vol_50 =data['Vol']

data = np.load('vertically-integrated-model-cold-firn-old_100by2_T-100C.npz')
t_100=data['t'] ;h_sol_100 =data['h_sol']; r_max_100 =data['r_max']; h_max_100 =data['h_max']; Vol_100 =data['Vol']



T=[0,-10,-20,-50,-100]
time_step = 10
koverk1=[1,0.9751428571428573,0.9502857142857141,0.8757142857142858,0.7514285714285711]
beta_arr= [0.333325  , 0.33170557, 0.33003462, 0.32468261, 0.31440889]
beta_expt = np.array([np.log(r_max_0[-1]/r_max_0[-time_step])/np.log(tyear[-1]/tyear[-time_step]), np.log(r_max_10[-1]/r_max_10[-time_step])/np.log(tyear[-1]/tyear[-time_step]),np.log(r_max_20[-1]/r_max_20[-time_step])/np.log(tyear[-1]/tyear[-time_step]) , np.log(r_max_50[-1]/r_max_50[-time_step])/np.log(tyear[-1]/tyear[-time_step]), np.log(r_max_100[-1]/r_max_100[-time_step])/np.log(tyear[-1]/tyear[-time_step])])
################################################################


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
plt.subplot(2,2,1)
#plt.loglog(tyear,h_max_analy,'r-',linewidth=3)
plt.plot(tyear,h_max_0,'b-',linewidth=5,label='$0^oC$',color=red,mfc='none',markersize=10)
plt.plot(tyear,h_max_10,'b-',linewidth=5,label='$-10^oC$',color=brown,mfc='none',markersize=10)
plt.plot(tyear,h_max_20,'b-',linewidth=5,label='$-20^oC$',color=purple,mfc='none',markersize=10)
plt.plot(tyear,h_max_50,'b-',linewidth=5,label='$-50^oC$',color=blue,mfc='none',markersize=10)
plt.plot(tyear,h_max_100,'b-',linewidth=5,label='$-100^oC$',color=green,mfc='none',markersize=10)
plt.ylabel(r'$h_{max} [m]$')
#plt.ylim([-0.6,0.05])
plt.legend(loc='best',framealpha=0.0)
#plt.axis('scaled')

plt.subplot(2,2,2)
#plt.loglog(tyear,x_max_analy,'r-',linewidth=3)
plt.plot(tyear,r_max_50,'b-',linewidth=5,color=blue,mfc='none',markersize=10)
plt.plot(tyear,r_max_50[-1]*(tyear/tyear[-1])**beta_arr[3],'k--',linewidth=2,mfc='none',markersize=10)
plt.plot(tyear,r_max_10,'b-',linewidth=5,color=brown,mfc='none',markersize=10)
plt.plot(tyear,r_max_10[-1]*(tyear/tyear[-1])**beta_arr[1],'k--',linewidth=2,mfc='none',markersize=10)
plt.plot(tyear,r_max_20,'b-',linewidth=5,color=purple,mfc='none',markersize=10)
plt.plot(tyear,r_max_20[-1]*(tyear/tyear[-1])**beta_arr[2],'k--',linewidth=2,mfc='none',markersize=10)
plt.plot(tyear,r_max_0,'b-',linewidth=5,color=red,mfc='none',markersize=10)
plt.plot(tyear,r_max_0[-1]*(tyear/tyear[-1])**beta_arr[0],'k--',linewidth=2,mfc='none',markersize=10)
plt.plot(tyear,r_max_100,'b-',linewidth=5,color=green,mfc='none',markersize=10)
plt.plot(tyear,r_max_100[-1]*(tyear/tyear[-1])**beta_arr[4],'k--',linewidth=2,mfc='none',markersize=10)
plt.xlabel(r'$t [yr]$')
plt.ylabel(r'$r_{max}$ [m]')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlabel(r'$t [yr]$')
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)

#We need to half the volume since we have 2 cells in Grid.dy directions
plt.subplot(2,2,3)
Vol = np.sum(h_sol,0)*Grid.dx*phi0
#plt.loglog(tyear,x_max_analy,'r-',linewidth=3)
plt.plot(tyear,Vol_50/2,'b-',linewidth=5,color=blue,mfc='none',markersize=10)
plt.plot(tyear,Vol_10/2,'b-',linewidth=5,color=brown,mfc='none',markersize=10)
plt.plot(tyear,Vol_20/2,'b-',linewidth=5,color=purple,mfc='none',markersize=10)
plt.plot(tyear,Vol_0/2,'b-',linewidth=5,color=red,mfc='none',markersize=10)
plt.plot(tyear,Vol_100/2,'b-',linewidth=5,color=green,mfc='none',markersize=10)
plt.xlabel(r'$t [yr]$')
plt.ylabel(r'$Q_{max} [m^3]$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
#plt.axis('scaled')
#plt.ylim([-0.1,0.7])
#plt.legend(loc='best',framealpha=0.0)
#plt.xlim([Grid.xmin, 10])
plt.subplots_adjust(wspace=0.0, hspace=0)
plt.tight_layout()
plt.savefig(f'../Figures/max_length_height_mound_cold_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)


plt.figure(figsize=(10,4),dpi=100)
t_arr = (np.array([0, 50, 500])).astype(int)
#for i in np.linspace(0,np.shape(h_sol)[1]-1,Num).round().astype(int):


for i in t_arr:
    ii = np.argwhere(i==t_arr)[0][0]+1
    #if i >0:
    Ratio = 1-((ii+1)/(len(t_arr)+1))
    Resulting_Color_blue = np.multiply((1 - Ratio),blue) + np.array([1,1,1]) * Ratio
    plt.plot(Grid.xc/1e3, np.transpose(h_sol_50[:,i].reshape(Grid.Nx,Grid.Ny))[0,:],color=Resulting_Color_blue,linewidth=5,label=r'-50$^o$C, %0.0f yr'%(t[i]/yr2s))
for i in t_arr:
    ii = np.argwhere(i==t_arr)[0][0]+1
    #if i >0:
    Ratio = 1-((ii+1)/(len(t_arr)+1))
    Resulting_Color_red  = np.multiply((1 - Ratio),red) + np.array([1,1,1]) * Ratio
    plt.plot(Grid.xc/1e3, np.transpose(h_sol_0[:,i].reshape(Grid.Nx,Grid.Ny))[0,:],color=Resulting_Color_red,linewidth=5,linestyle='--',label=r'0$^o$C, %0.0f yr'%(t[i]/yr2s))
plt.ylabel(r'$h$ [m]')
plt.xlabel(r'$x$ [m]')
plt.tight_layout()
plt.legend(loc='best',ncol=2)
plt.savefig(f'../Figures/{simulation_name}_{tilt_angle}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h_combined_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)



