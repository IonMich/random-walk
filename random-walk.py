#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:37:43 2018

@author: yannis
"""
import numpy as np
import matplotlib.pyplot as plt


def photonParallelOdysseys(radius=1,nWalks=1,printing=False):
    """
    Generates simultaneously random walks (nWalks number of walks) that terminate when they reach the radius (given as input in meters)
    Returns the exit times in seconds
    NOTE: Since the RMS distance at n steps of a Random Walk is independent of the number of dimensions
    we can use a 2D random walk to estimate the time needed for the photon to escape the 3D interior of the Sun
    """
    radiusSq = radius**2
    exitTimesArray = np.zeros(nWalks) 
        
    ## Initialize the position of the photon to be at the center of the Sun
    photonXYArray = np.zeros((2,nWalks) ,dtype =np.float64 )
    nStepsCompleted = 0
    walksRemaining = nWalks
    while (np.sum(photonXYArray*photonXYArray, axis=0) < radiusSq).any():

        ## creating a step in a random direction
        thetaStep = np.random.uniform(0 , 2*np.pi,walksRemaining)
        stepArray = photonSolarFreePath * np.array([np.cos(thetaStep) , np.sin(thetaStep)])
        ## assigning new location
        photonXYArray += stepArray
        nStepsCompleted += 1
        
        while (np.sum(photonXYArray*photonXYArray, axis=0) > radiusSq).any():
            
            indexToDelete = (np.sum(photonXYArray*photonXYArray, axis=0) > radiusSq).nonzero()[0][0]
            ##remove a walk that is completed:
            photonXYArray = np.delete(photonXYArray,indexToDelete,1)
            
            exitTimesArray[nWalks - walksRemaining] = nStepsCompleted * timeOfStep
            ## printing once in a while:
            conditionToPrint = (walksRemaining%1000==1 and nWalks>5000) or (walksRemaining%200==1 and nWalks<=5000 and nWalks>1000) or (walksRemaining%20==1 and nWalks<=1000 and nWalks>100) or (walksRemaining%5==1 and nWalks<=100 and nWalks>10) or (nWalks<=10)
            if printing==True and conditionToPrint:
                print("\nSteps needed to exit: {}\nTime in seconds needed to exit: {}".format(nStepsCompleted,exitTimesArray[nWalks - walksRemaining]))
                print("{} walks remaining".format(walksRemaining-1))
            walksRemaining -= 1

    
    return exitTimesArray



def calculatingTimesParallel(radius=1,nWalks=5):
    """
    Finding the time (in secs) and standard deviation(in secs) needed for to exit the Sun given a radius
    using nWalks trials
    """

    print("\nRandom Walks inside a circle of radius {} meters:\n".format(radius))
    timesToExit = photonParallelOdysseys(radius=radius,nWalks=nWalks,printing=True)
    timeAverage = np.sum(timesToExit)/len(timesToExit)
    rmsError = np.sqrt( np.sum((timesToExit-timeAverage)**2) / len(timesToExit) )
    
    return timeAverage, rmsError



photonSolarFreePath = 4E-3  # mean free path of photons inside the Sun in meters
speedOfLight = 3E8 # speed of light in meters per second
timeOfStep = photonSolarFreePath / speedOfLight
fromSecToYears = 1/(86400*365)

solarRadius = 7E8 # the radius of the Sun in meters
        
if __name__ == "__main__":
    
    
    ## finding the time (in secs) needed for 0.7 meters    
    time0_7 = calculatingTimesParallel(radius=0.7,nWalks=10000)  
    print("\nAverage time in seconds needed to exit 0.7 meters: {}\nStandard Deviation of realizations in seconds: {}".format(time0_7[0],time0_7[1]))
    
    ## finding the time (in secs) needed for 7.0 meters
    time2_0 = calculatingTimesParallel(radius=2.0,nWalks=200)
    print("\nAverage time in seconds needed to exit 2.0 meters: {}\nStandard Deviation of realizations in seconds: {}".format(time2_0[0],time2_0[1]))
    
    ## finding the time (in secs) needed for 7.0 meters
    time7_0 = calculatingTimesParallel(radius=7.0,nWalks=50)
    print("\nAverage time in seconds needed to exit 7.0 meters: {}\nStandard Deviation of realizations in seconds: {}".format(time7_0[0],time7_0[1]))
    
    
    ## Generating the plots
    radiiArray = np.array([0,0.7,2.0,7.0])
    timesArray = fromSecToYears * np.array([0,time0_7[0],time2_0[0],time7_0[0]])
    p2 = np.poly1d(np.polyfit(radiiArray, timesArray, 2))
    

    _ , (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6),num='Photon Exit Times')

    ax1.set_title("Calculation of exit times")
    ax1.set_xlabel("Radius (meters)")
    ax1.set_ylabel("Exit Time (years)")
    ax1.plot(radiiArray,timesArray,"*k",label='RW Estimate')
    xpNarrow = 7 * np.logspace(-3, 0, 500)
    ax1.plot(xpNarrow,p2(xpNarrow),linestyle='-',label='Quadratic Fit')
    ax1.legend()
    ax1.grid()
    
    
    ax2.set_title("Estimation of the actual Escape Time of Solar Photons")
    ax2.set_xlabel("Radius (meters)")
    ax2.set_ylabel("Exit Time (years)")    
    xpWide = 7 * np.logspace(-1, 8, 500)
    ax2.plot(xpWide,p2(xpWide),linestyle='-',label='Quadratic Fit')
    ax2.axvline(x=solarRadius,linestyle='--',color='k')
    ax2.axhline(p2(solarRadius),linestyle='--',color='k')
    ax2.plot(solarRadius,p2(solarRadius),'or',label='True-R Estimate')    
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.show()
    
    print("\nUsing a quadratic fit, we estimete the mean escape time to be {:.1e} years".format( p2(solarRadius) ) )
    print("\nThe quadratic fit is justified by the fact that the average RMS distance from the origin of a random walk with step size=1 after N steps is sqrt(N).")
    quadraticCoeffTheory = 1/photonSolarFreePath/speedOfLight*fromSecToYears
    print("Incidenticaly, this means that the coefficient of radius^2 in our parabola should be 1/photonSolarFreePath/speedOfLight = {:.2e} years/m**2".format(quadraticCoeffTheory))
    print("Indeed the fit gives that this coefficient is: {:.2e} years/m**2".format(p2[2]))
    
    
    
    
    