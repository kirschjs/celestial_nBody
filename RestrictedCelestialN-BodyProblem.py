import scipy as sci
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

#Gravitational constant
G=7e-5
#simtime
d=60*60*24
y=365*d
T = 200*d
h = d
a = int(T/h)
t = np.zeros(a)

#Zwangsbedinungen: Kreisbahnen mit Radius r und winkelgeschw omega Masse und Anfangswinkel/offset
Rad = np.array([57.9e9,108e9,149e9,228e9])
Omega = np.array([2*math.pi/7600000,2*math.pi/19400000,2*math.pi/31500000,2*math.pi/594000000])
Mass = np.array([3.3e23,4.87e24,5.97e24,6.42e23])
alpha_t0 = np.array([0,0,0,0])

#plotting planeten 1-4
x_1=np.zeros((a))
x_2=np.zeros((a))
x_3=np.zeros((a))
x_4=np.zeros((a))
y_1=np.zeros((a))
y_2=np.zeros((a))
y_3=np.zeros((a))
y_4=np.zeros((a))

def Schwerpunktkoordinaten(r1,r2,r3,r4,r5,m,M):
    masses=[M[0],M[1],M[2],M[3],m]
    position=[r1,r2,r3,r4,r5]
    R=0
    for i in range(0,len(masses)-1):
        R=R+masses[i]*position[i]
    return R/len(masses)

#ort der festgelegten Bahnen
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
r[0]=[10000000000,0,1000000000] #r0
p[0]=[0,0,0] #p0

#simulation data
x_5=np.zeros((a))
x_5[0]=r[0,0]
y_5=np.zeros((a))
y_5[0]=r[0,1]

#dp/dt
def fp(r,p,t):
    dp=0
    for i in range(0,len(Mass)-1):
        dp = dp -G*Mass[i]*m*(r-ri(i,t))/(np.linalg.norm(r-ri(i,t))**3)
    return dp

#dr/dt
def fr(r,p,t):
    dp=p/m
    return dp

#runge kutta 4 vectors
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

#main
for i in range(1,a):

    rn,pn = rkvec(r[i-1],p[i-1],fr,fp,t[i-1])

    t[i]=t[i-1]+h

    r[i]=rn
    x_5[i]=rn[0]
    y_5[i]=rn[1]

    p[i]=pn

    r1=ri(0,t[i])
    r2=ri(1,t[i])
    r3=ri(2,t[i])
    r4=ri(3,t[i])

    R=Schwerpunktkoordinaten(r1,r2,r3,r4,rn,m,Mass)
    x_5[i]=rn[0]
    y_5[i]=rn[1]
    x_1[i]=r1[0]
    x_2[i]=r2[0]
    x_3[i]=r3[0]
    x_4[i]=r4[0]
    y_1[i]=r1[1]
    y_2[i]=r2[1]
    y_3[i]=r3[1]
    y_4[i]=r4[1]

fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)
ax.plot(x_1,y_1,color="mediumblue")
ax.plot(x_2,y_2,color="red")
ax.plot(x_3,y_3,color="gold")
ax.plot(x_5,y_5,color="lightgreen")
ax.plot(x_5,y_5,color="grey")
ax.scatter(x_1,y_1,color="darkblue",marker="o",s=80,label="Star 1")
ax.scatter(x_2,y_2,color="darkred",marker="o",s=80,label="Star 2")
ax.scatter(x_3,y_3,color="goldenrod",marker="o",s=80,label="Star 3")
ax.scatter(x_4,y_4,color="green",marker="o",s=80,label="Star 4")
ax.scatter(x_5,y_5,color="black",marker="o",s=80,label="Star 5")
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a 3-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)


#Animate the orbits of the three bodies


#Make the figure 
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)
plt.xlim(-5*np.max(x_4), 5*np.max(x_4))
plt.ylim(-10*np.max(y_4), 10*np.max(y_4))

#Set initial marker for planets, that is, blue,red and green circles at the initial positions
head1=[ax.scatter(x_1[0],y_1[0],color="darkblue",marker="o",s=80,label="Star 1")]
head2=[ax.scatter(x_2[0],y_2[0],color="darkred",marker="o",s=80,label="Star 2")]
head3=[ax.scatter(x_3[0],y_3[0],color="goldenrod",marker="o",s=80,label="Star 3")]
head4=[ax.scatter(x_4[0],y_4[0],color="green",marker="o",s=80,label="Star 4")]
head5=[ax.scatter(x_5[0],y_5[0],color="black",marker="o",s=80,label="Star 5")]


#Create a function Animate that changes plots every frame (here "i" is the frame number)
def Animate(i,head1,head2,head3,head4,head5):
    #Remove old markers
    head1[0].remove()
    head2[0].remove()
    head3[0].remove()
    head4[0].remove()
    head5[0].remove()
    
    #Plot the orbits (every iteration we plot from initial position to the current position)
    trace1=ax.plot(x_1[i],y_1[i],color="mediumblue")
    trace2=ax.plot(x_2[i],y_2[i],color="red")
    trace3=ax.plot(x_3[i],y_3[i],color="gold")
    trace4=ax.plot(x_4[i],y_4[i],color="lightgreen")
    trace5=ax.plot(x_5[i],y_5[i],color="grey")
    
    #Plot the current markers
    head1[0]=ax.scatter(x_1[i-1],y_1[i-1],color="darkblue",marker="o",s=100)
    head2[0]=ax.scatter(x_2[i-1],y_2[i-1],color="darkred",marker="o",s=100)
    head3[0]=ax.scatter(x_3[i-1],y_3[i-1],color="goldenrod",marker="o",s=100)
    head4[0]=ax.scatter(x_4[i-1],y_4[i-1],color="green",marker="o",s=100)
    head5[0]=ax.scatter(x_5[i-1],y_5[i-1],color="black",marker="o",s=100)
    return trace1,trace2,trace3,trace4,trace5,head1,head2,head3,head4,head5
#Some beautifying
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a 3-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)


#If used in Jupyter Notebook, animation will not display only a static image will display with this command
# anim_2b = animation.FuncAnimation(fig,Animate_2b,frames=1000,interval=5,repeat=False,blit=False,fargs=(h1,h2))


#Use the FuncAnimation module to make the animation
repeatanim=animation.FuncAnimation(fig,Animate,frames=800,interval=10,repeat=False,blit=False,fargs=(head1,head2,head3,head4,head5))
plt.show()