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
sys.path.insert(1, '../../solver')

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
simulation_name ='vertically-integrated-model-cold-firn-old_2D'
L = 1000   #Length of the glacier (m)
tmax = 10*365.25*day2s #10 years max #Maximum time (s)
deg2rad = np.pi/180

##Problem parameters
R = 0#0.048/day2s #Recharge (m/s)
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
T  = -30 # Temperature of the aquifer [C]
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
Grid.ymin = 0; Grid.ymax = L; Grid.Ny = 100

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
h[Xc_col**2 + Yc_col**2<x_right**2] = h_top #10m high


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
    
    if (i+1)%(Nt/500)==0:
        print(i+1,time/day2s, 'days')
        t = np.append(t,time)
        h_sol = np.hstack([h_sol,h])

        ##########################################################################################          
        #maximum values for analysis
        arr = h[0:Grid.N-1:Grid.Ny]>1e-6
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
Vol = np.sum(h_sol,0)*Grid.dx*phi0*Grid.dy
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


plt.subplots_adjust(wspace=0.0, hspace=0)
plt.tight_layout()
plt.savefig(f'../Figures/max_length_height_mound_cold_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)



######################################################################
#Saving the data
######################################################################
np.savez(f'{simulation_name}_{Grid.Nx}by{Grid.Ny}_T{T}C.npz', t=t,h_sol=h_sol,r_max=r_max,h_max=h_max,dx = Grid.dx,Vol=Vol)



######################################################################
#for loading data
######################################################################
data = np.load(f'{simulation_name}_{Grid.Nx}by{Grid.Ny}_T{T}C.npz')
t=data['t']
h_sol =data['h_sol']
r_max =data['r_max']
h_max =data['h_max']



plt.figure(figsize=(10,4),dpi=100)
Num = 6
for i in np.linspace(0,np.shape(h_sol)[1]-1,Num).round().astype(int):
    if i >0:
        plt.plot(Grid.xc/1e3, np.transpose(h_sol[:,i].reshape(Grid.Nx,Grid.Ny))[0,:],color=blue,linewidth=5,alpha=((i+10)/(np.shape(h_sol)[1]-1+20)),label=r'%0.1f yrs'%(t[i]/yr2s))
plt.ylabel(r'$h$ [m]')
plt.xlabel(r'$x$ [m]')
plt.tight_layout()
plt.legend(loc='best',ncol=3)
plt.savefig(f'../Figures/{simulation_name}_{0}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h_combined_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)



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
plt.savefig(f'../Figures/{simulation_name}_{0}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h_combined_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)




plt.figure(figsize=(8,6),dpi=100)
t_arr = (np.array([0, 10, 50, 100])).astype(int)
#for i in np.linspace(0,np.shape(h_sol)[1]-1,Num).round().astype(int):
for ii in [-1]:
    #for i in t_arr:
    #ii = np.argwhere(i==t_arr)[0][0]+1
    plt.contourf(Xc/1e3,Yc/1e3, np.transpose(h_sol[:,ii].reshape(Grid.Nx,Grid.Ny))) #,label=r'%0.0f yrs'%(t[i]/yr2s)
plt.ylabel(r'$y$ [m]')
plt.xlabel(r'$x$ [m]')
plt.colorbar()
plt.tight_layout()
plt.axis('scaled')
plt.legend(loc='best',ncol=2)
plt.savefig(f'../Figures/{simulation_name}_{0}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h_combined_T{T}C_old_contour.pdf',bbox_inches='tight', dpi = 600)



'''
#New Contour plot
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(h_sol) # frame number of the animation from the saved file
tyear = t/yr2s
def update_plot(frame_number, zarray, plot,t):
    plt.xlabel(r'$x$ [km]')
    plt.ylabel(r'$y$ [km]')
    #plt.xlim([Grid.xmin, Grid.xmax/1e3])
    #plt.ylim([0, np.max(h_sol)])
    plot[0] = plt.contourf(Xc/1e3,Yc/1e3, np.transpose(h_sol[:,frame_number].reshape(Grid.Nx,Grid.Ny)),vmin=np.min(h_sol),vmax=np.max(h_sol),levels=100)
    plt.title("t= %0.2f years" % tyear[frame_number],loc = 'center', fontsize=18)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.axis('scaled')
    plt.clim(np.min(h_sol),np.max(h_sol))


fig = plt.figure(figsize=(10,10) , dpi=100)
plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$y$ [m]')
#plt.xlim([Grid.xmin, Grid.xmax/1e3])
#plt.ylim([0, np.max(h_sol)])
plot = [plt.contourf(Xc/1e3,Yc/1e3, np.transpose(h_sol[:,0].reshape(Grid.Nx,Grid.Ny)),vmin=np.min(h_sol),vmax=np.max(h_sol),levels=100) ]
plt.title("t= %0.2f years" % tyear[0],loc = 'center', fontsize=18)
plt.axis('scaled')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(h_sol)
mm.set_clim(np.min(h_sol),np.max(h_sol))
clb = plt.colorbar()#mm, pad=0.1)
clb.set_label(r'$h$ [m]', labelpad=-3,x=-3, y=1.13, rotation=0)
    
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(h_sol[:,:], plot[:],tyear[:]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{0}degree__tf{t[frn-1]}T_{T}C.mov", writer='ffmpeg', fps=30)
'''






################################################################
#Analysis script
################################################################

tyear = t/yr2s
data = np.load('vertically-integrated-model-cold-firn-old_2D_100by100_T0C.npz')
t_0=data['t'] ;h_sol_0 =data['h_sol']; r_max_0 =data['r_max']; h_max_0 =data['h_max']; Vol_0 =data['Vol']

data = np.load('vertically-integrated-model-cold-firn-old_2D_100by100_T-10C.npz')
t_10=data['t'] ;h_sol_10 =data['h_sol']; r_max_10 =data['r_max']; h_max_10 =data['h_max']; Vol_10 =data['Vol']

data = np.load('vertically-integrated-model-cold-firn-old_2D_100by100_T-30C.npz')
t_30=data['t'] ;h_sol_30 =data['h_sol']; r_max_30 =data['r_max']; h_max_30 =data['h_max']; Vol_30 =data['Vol']

data = np.load('vertically-integrated-model-cold-firn-old_2D_100by100_T-50C.npz')
t_50=data['t'] ;h_sol_50 =data['h_sol']; r_max_50 =data['r_max']; h_max_50 =data['h_max']; Vol_50 =data['Vol']

data = np.load('vertically-integrated-model-cold-firn-old_2D_100by100_T-100C.npz')
t_100=data['t'] ;h_sol_100 =data['h_sol']; r_max_100 =data['r_max']; h_max_100 =data['h_max']; Vol_100 =data['Vol']


import pandas as pd
from scipy.interpolate import interp1d
# Load the CSV and give names to columns
file_path = "./cylindrical_betavskappa_ratio.csv"
df = pd.read_csv(file_path, header=None, names=["beta", "kappa_ratio"])

# Build interpolation function (linear)
interp_func = interp1d(df["beta"],df["kappa_ratio"], kind="linear", fill_value="extrapolate")

# Example: evaluate interpolation on a fine grid
kappa_smooth = np.linspace(df["beta"].min(), df["beta"].max(), 5000)
beta_smooth = interp_func(kappa_smooth)


TT=[0,-10,-30,-50,-100]
time_step = 150
koverk1=[1,0.9751428571428573,0.9254285714285713,0.8757142857142858,0.7514285714285711]
#beta_arr= [0.333325  , 0.33170557, 0.33003462, 0.32468261, 0.31440889]
#beta_expt = np.array([np.log(r_max_0[-1]/r_max_0[-time_step])/np.log(tyear[-1]/tyear[-time_step]), np.log(r_max_10[-1]/r_max_10[-time_step])/np.log(tyear[-1]/tyear[-time_step]),np.log(r_max_30[-1]/r_max_30[-time_step])/np.log(tyear[-1]/tyear[-time_step]) , np.log(r_max_50[-1]/r_max_50[-time_step])/np.log(tyear[-1]/tyear[-time_step]), np.log(r_max_100[-1]/r_max_100[-time_step])/np.log(tyear[-1]/tyear[-time_step])])
beta_expt = interp_func(koverk1)
################################################################


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
plt.subplot(2,2,1)
#plt.loglog(tyear,h_max_analy,'r-',linewidth=3)
plt.plot(tyear,h_max_0,'b-',linewidth=5,label='$0^o$C',color=red,mfc='none',markersize=10)
plt.plot(tyear,h_max_10,'b-',linewidth=5,label='$-10^o$C',color=brown,mfc='none',markersize=10)
plt.plot(tyear,h_max_30,'b-',linewidth=5,label='$-30^o$C',color=blue,mfc='none',markersize=10)
plt.plot(tyear,h_max_50,'b-',linewidth=5,label='$-50^o$C',color=purple,mfc='none',markersize=10)
plt.plot(tyear,h_max_100,'b-',linewidth=5,label='$-100^o$C',color=green,mfc='none',markersize=10)
plt.ylabel(r'$h_{max} [m]$')
#plt.ylim([-0.6,0.05])
plt.xlabel(r'$t [yr]$')
plt.legend(loc='best',framealpha=0.0)
#plt.axis('scaled')

plt.subplot(2,2,2)
#plt.loglog(tyear,x_max_analy,'r-',linewidth=3)
plt.plot(tyear,r_max_50,'b-',linewidth=5,color=purple,mfc='none',markersize=10)
plt.plot(tyear,r_max_50[-1]*(tyear/tyear[-1])**beta_expt[3],'k--',linewidth=2,mfc='none',markersize=10)
plt.plot(tyear,r_max_10,'b-',linewidth=5,color=brown,mfc='none',markersize=10)
plt.plot(tyear,r_max_10[-1]*(tyear/tyear[-1])**beta_expt[1],'k--',linewidth=2,mfc='none',markersize=10)
plt.plot(tyear,r_max_30,'b-',linewidth=5,color=blue,mfc='none',markersize=10)
plt.plot(tyear,r_max_30[-1]*(tyear/tyear[-1])**beta_expt[2],'k--',linewidth=2,mfc='none',markersize=10)
plt.plot(tyear,r_max_0,'b-',linewidth=5,color=red,mfc='none',markersize=10)
plt.plot(tyear,r_max_0[-1]*(tyear/tyear[-1])**beta_expt[0],'k--',linewidth=2,mfc='none',markersize=10)
plt.plot(tyear,r_max_100,'b-',linewidth=5,color=green,mfc='none',markersize=10)
plt.plot(tyear,r_max_100[-1]*(tyear/tyear[-1])**beta_expt[4],'k--',linewidth=2,mfc='none',markersize=10)
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
Vol = np.sum(h_sol,0)*Grid.dx*Grid.dy*phi0
#plt.loglog(tyear,x_max_analy,'r-',linewidth=3)
plt.plot(tyear,Vol_50/2,'b-',linewidth=5,color=purple,mfc='none',markersize=10)
plt.plot(tyear,Vol_10/2,'b-',linewidth=5,color=brown,mfc='none',markersize=10)
plt.plot(tyear,Vol_30/2,'b-',linewidth=5,color=blue,mfc='none',markersize=10)
plt.plot(tyear,Vol_0/2,'b-',linewidth=5,color=red,mfc='none',markersize=10)
plt.plot(tyear,Vol_100/2,'b-',linewidth=5,color=green,mfc='none',markersize=10)
plt.xlabel(r'$t [yr]$')
plt.ylabel(r'$Q [m^3]$')

plt.subplot(2,2,4)
Vol = np.sum(h_sol,0)*Grid.dx*phi0*Grid.dy
plt.plot(kappa_smooth,beta_smooth,'k--',linewidth=2.5)
plt.plot(koverk1[0],beta_expt[0],'ro',color=red,markersize=15)
plt.plot(koverk1[1],beta_expt[1],'ro',color=brown,markersize=15)
plt.plot(koverk1[2],beta_expt[2],'ro',color=blue,markersize=15)
plt.plot(koverk1[3],beta_expt[3],'ro',color=purple,markersize=15)
plt.plot(koverk1[4],beta_expt[4],'ro',color=green,markersize=15)
plt.xlim([0.6,1.02])
plt.ylim([0.22,0.25])
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\kappa$ / $\kappa_1$')


plt.subplots_adjust(wspace=0.5, hspace=0)
plt.tight_layout()
plt.savefig(f'../Figures/max_length_height_mound_cold_T{T}C_old2D.pdf',bbox_inches='tight', dpi = 600)





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
plt.savefig(f'../Figures/{simulation_name}_{0}degree_{Grid.Nx}by{Grid.Ny}_t{t[-1]}_h_combined_T{T}C_old.pdf',bbox_inches='tight', dpi = 600)




#contour plots

tyear = t/yr2s
data = np.load('vertically-integrated-model-cold-firn-old_2D_100by100_T0C.npz')
t_0=data['t'] ;h_sol_0 =data['h_sol']; r_max_0 =data['r_max']; h_max_0 =data['h_max']; Vol_0 =data['Vol']


#combined contours
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,8))
plt.subplot(3,2,1)
plt.ylabel(r'$y$ [km]')
plot =plt.contourf(Xc/1e3,Yc/1e3, np.transpose(h_sol_0[:,0].reshape(Grid.Nx,Grid.Ny)),vmin=np.min(h_sol_0),vmax=np.max(h_sol_0),levels=100) 
#plt.title("t= %0.2f years" % tyear[0],loc = 'center', fontsize=18)
plt.axis('scaled')
plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)


plt.subplot(3,2,3)
plt.ylabel(r'$y$ [km]')
plot =plt.contourf(Xc/1e3,Yc/1e3, np.transpose(h_sol_0[:,0].reshape(Grid.Nx,Grid.Ny)),vmin=np.min(h_sol_0),vmax=np.max(h_sol_0),levels=100) 
#plt.title("t= %0.2f years" % tyear[0],loc = 'center', fontsize=18)
plt.axis('scaled')
plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)



plt.subplot(3,2,5)
plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$y$ [km]')
plot =plt.contourf(Xc/1e3,Yc/1e3, np.transpose(h_sol_0[:,0].reshape(Grid.Nx,Grid.Ny)),vmin=np.min(h_sol_0),vmax=np.max(h_sol_0),levels=100) 
#plt.title("t= %0.2f years" % tyear[0],loc = 'center', fontsize=18)
plt.axis('scaled')
plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
plt.subplots_adjust(wspace=0.0, hspace=0)
plt.tight_layout()
plt.savefig(f'../Figures/max_length_height_mound_cold_T{T}C_old_combined.pdf',bbox_inches='tight', dpi = 600)



import matplotlib.colors as mcolors
ii = [0,0,50,50,500,500]
# 3x2 contour plots with tight layout and titled colorbar
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7.2, 10), sharex=True, sharey=True)

# Force values above 3 to saturate at the top color
norm = mcolors.Normalize(vmin=-2, vmax=1)
# Define common contour levels
levels = np.linspace(-2, 1, 101)




import matplotlib.colors as mcolors
for i, ax in enumerate(axes.flat):
    if i%2 == 0:
        c = ax.contourf(
        Xc/1e3, Yc/1e3,
        np.transpose(np.log10(h_sol_0)[:, ii[i]].reshape(Grid.Nx, Grid.Ny)),
        levels=levels, cmap="Blues", norm=norm, edgecolor="none", antialiased=False,rasterized=True, linewidths=0, ls=None
        )
        c.set_edgecolor("face")
    else:
        c = ax.contourf(
        Xc/1e3, Yc/1e3,
        np.transpose(np.log10(h_sol_30)[:, ii[i]].reshape(Grid.Nx, Grid.Ny)),
        levels=levels, cmap="Blues", norm=norm, edgecolor="none", antialiased=False,rasterized=True, linewidths=0, ls=None
        )
        
        # This is the fix for the white lines between contour levels
        c.set_edgecolor("face")
    
    ax.set_aspect("equal")

plt.xticks([0, 0.25,0.5])
plt.yticks([0, 0.25,0.5])
# Add axis labels only on outer edges
for ax in axes[:, 0]:
    ax.set_ylabel(r"$y$ [km]")
for ax in axes[-1, :]:
    ax.set_xlabel(r"$x$ [km]")

# Tilt x-axis tick labels on bottom row
for ax in axes[-1, :]:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

#plt.tight_layout()

# Shared colorbar with title
# Shared colorbar with label on top
ticks = [-2, -1, 0, 1]   # corresponds to 0.01, 0.1, 1, 10
cbar = fig.colorbar(
    c, ax=axes.ravel().tolist(),
    orientation="vertical", fraction=0.025, pad=0.1, ticks=ticks
)

# Put label on top
cbar.ax.set_title("h [m]", pad=10)

# Format ticks as powers of 10
cbar.ax.set_yticklabels([r"10$^{%s}$" % t for t in ticks])

# Show ticks inside plots but remove labels
for ax in axes.flat:
    ax.tick_params(
        which="both",
        direction="in",   # ticks inside
        top=True, right=True,  # ticks on all sides
    )
#plt.tight_layout()
plt.tight_layout()
plt.xlim([0, 0.62])
plt.ylim([0, 0.62])
# Reduce space between plots and colorbar
plt.subplots_adjust(wspace=0.00, hspace=0.0, right=0.81)  # right leaves room for colorbar
#set(gcf, 'Renderer', 'opengl');
plt.savefig(
    f"../Figures/max_length_height_mound_cold_T{T}C_old_combined.pdf", dpi=600, transparent=True)
plt.show()