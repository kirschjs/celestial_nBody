import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import math

#Gravitational constant
G=7e-11
#simtime
T = 1000000000000
h = 10000000
a = int(T/h)
t = np.zeros(a)

#Zwangsbedinungen: Kreisbahnen mit Radius r und winkelgeschw omega
Rad = np.array([57.9e9,108e9,149e9,228e9])
Omega = np.array([2*math.pi/7600000,2*math.pi/19400000,2*math.pi/31500000,2*math.pi/594000000])
Mass = np.array([3.3e23,4.87e24,5.97e24,6.42e23])
alpha_t0 = np.array([0,math.pi/2,math.pi,3*math.pi/2])

def ri(i,t):
    x = Rad[i]*math.cos(Omega[i]*t+alpha_t0[i])
    y = Rad[i]*math.sin(Omega[i]*t+alpha_t0[i])
    z=0
    return  np.array([x,y,z])

#mass
m=1000

#r und p des 5. Planeten
r = np.zeros((a,3))
p = np.zeros((a,3))
r[0]=[10000000000,0,0] #r0
p[0]=[0,0,0] #p0

#simulation data
x=np.zeros((a))
x[0]=r[0,0]
y=np.zeros((a))
y[0]=r[0,1]


def fp(r,p,t):
    dp=0
    for i in range(0,len(Mass)-1):
        dp = dp -G*Mass[i]*m*(r-ri(i,t))/(np.linalg.norm(r-ri(i,t))**3)
    return dp

def fr(r,p,t):
    dp=p/m
    return dp

def rkvec(r0,p0,fr,fp,t):
    k1 = fr(r0,p0,t)
    l1 = fp(r0,p0,t)
    k2 = fr(r0+h*k1/2,p0+h*l1/2,t+h/2)
    l2 = fp(r0+h*k1/2,p0+h*l1/2,t+h/2)
    k3 = fr(r0+h*k2/2,p0+h*l2/2,t+h/2)
    l3 = fp(r0+h*k2/2,p0+h*l2/2,t+h/2)
    k4 = fr(r0+h*k3,p0+h*l3,t+h)
    l4 = fp(r0+h*k3,p0+h*l3,t+h)
    k = h*(k1+2*k2+2*k3+k4)/6
    l = h*(l1+2*l2+2*l3+l4)/6
    r_new = r0+k
    p_new = p0+l
    return r_new,p_new


for i in range(1,a):

    rn,pn = rkvec(r[i-1],p[i-1],fr,fp,t[i-1])

    t[i]=t[i-1]+h

    r[i]=rn
    x[i]=rn[0]
    y[i]=rn[1]

    p[i]=pn
    #print(i/a)
    
plt.plot(t,x)
plt.show()
