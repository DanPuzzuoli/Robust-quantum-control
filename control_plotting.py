#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:50:31 2018

@author: Daniel Puzzuoli

Description: 
    Currently very rudimentary, but a place to put some plotting
    functions for pulses as required
"""
import matplotlib.pyplot as plt
from numpy import linspace,append, empty
from control_system import control_system
from evolve_system import evolve_system
from objective_functions import grape_objective

def step_plot(amps, dt, ybound = None):  
    
    N = len(amps)
    amps = append(amps.copy(),[amps[-1]],axis=0)
    
    plt.step(linspace(start = 0, stop=N*dt, num=N+1),amps, where = 'post')
    plt.show()
    
def target_robustness_1d(system, amps,dt,Utarget, gvar, vals):
    # initialize the fidelities
    fidelities = empty(len(vals))
    
    for k in range(len(vals)):
        c_sys = control_system(system.drift + vals[k]*gvar, system.control_generators.copy())
        fidelities[k] = grape_objective(Utarget, evolve_system(c_sys,amps,dt))/4
    
    plt.plot(vals, fidelities)
    plt.show()