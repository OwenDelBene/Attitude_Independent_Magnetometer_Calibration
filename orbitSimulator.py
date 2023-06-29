import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


class Constants:
    MU = 3.986e14 #m 
    RE = 6.371e6 # m
def ode(t, state):
    r = state[:3]
    a = -Constants.MU* r / np.linalg.norm(r) ** 3
    

    return np.array([state[3], state[4], state[5], a[0], a[1], a[2]])



def rk4(f, t, state, stepsize):
    k1 = f(t,state)
    k2 = f(t + .5*stepsize, state + .5*k1 * stepsize)
    k3 = f(t + .5 *stepsize, state + .5 * k2 * stepsize)
    k4 = f(t +stepsize, state + k3*stepsize)

    return state + stepsize/ 6.0 * (k1 + 2*k2 + 2*k3 + k4)


def circularVelocity(radius):
    return np.sqrt(Constants.MU / radius)

if __name__ == "__main__":
    #import igrf

    #mag = igrf.igrf('2010-07-12', glat=65, glon=-148, alt_km=100)
    #print(mag)

    initalAltitude = 400e3 #m
    initalVelocity = circularVelocity(initalAltitude + Constants.RE) #m/s
    initialState = [initalAltitude + Constants.RE, 0, 0, 0, initalVelocity, 0]

    ti = 0
    dt = 1    #seconds
    tf = 5600 #seconds
    states = np.empty((int(tf/dt), 6))
    states[0] = initialState
    for t in np.arange(ti,tf-dt,dt):
        states[t+1] = rk4(ode, t, states[t], dt)

    x = states[:,0]
    y = states[:,1]
    plt.plot(x,y)
    plt.show()
