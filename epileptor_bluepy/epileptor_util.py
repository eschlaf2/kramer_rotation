# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:33:41 2013.

Utility functions for calcium-based STDP using simplified calcium model as in
(Graupner and Brunel, 2012).

@author: Giuseppe Chindemi
@remark: Copyright © BBP/EPFL 2005-2016; All rights reserved.
         Do not distribute without further notice.
"""

# pylint: disable=R0914, R0912

import logging
import numpy as np
import integrators
from scipy import linalg as la
from scipy.special import erf  # NOQA
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARN)

# Note: having debug logging statements increases the run time by ~ 25%,
# because they exist in tight loops, and expand their outputs, even when
# debug is off, so we disable logging if possible.  Set this to true if
# verbose output is needed
LOGGING_DEBUG = True

plot = LOGGING_DEBUG

def logging_debug_vec(fmt, vec):
    '''log to debug a vector'''
    if LOGGING_DEBUG:
        logging.debug(fmt, ', '.join(map(str, vec)))


def logging_debug(*args):
    '''wrapper to log to debug a vector'''
    if LOGGING_DEBUG:
        logging.debug(*args)


# Parameters for epileptor (Jirsa et al., 2014)
param_epileptor = {
    'x0': -1.6,  # [s]
    'y0': 1.,
    'tau0': 2857,
    'tau1': 1.0,
    'tau2': 10.,
    'Irest1': 3.1,
    'Irest2': 0.45,
    'gamma': 1e-2,
    'x1_init': 0.,
    'y1_init': -5.,
    'z_init': 3.,
    'x2_init': 0.,
    'y2_init': 0.,
    'g_init': 0.,
    'noise_ensemble1': 25e-3,
    'noise_ensemble2': 25e-2}

class Protocol(object):

    """Protocol"""

    def __init__(self, x1_init=param_epileptor['x1_init'], y1_init=param_epileptor['y1_init'],
                 z_init=param_epileptor, prot_id=None):
        self.prot_id = prot_id

class Model(object):
    def __init__(self):
        self.augmented_state = []
    
    def generate_simulation(self,plot=plot,noisy=False):
        '''
            Simulates true and noisy trajectory based on previously
            defined model and parameter functions
            (Uses global vars)
            '''
        
        # Simulate model
        true_state = self.integrate_model()
        self.augmented_state = np.vstack((self.parameters*np.ones(\
                              (self.dims_params,self._num_samples)),true_state)) if self.dims_params > 0 \
                                else true_state
        self.dims_augmented_state = self.dims_params+self.dims_state_vars
        
        # Observation noise
        self.observation_sigmas = [0.2*0.2*np.var(self.observation_function(self.augmented_state))]
        observation_noise = np.diag(self.observation_sigmas)

        # Create noisy data from true trajectory
        self.data = self.observation_function(self.augmented_state)
        self.noisy_data = self.data + np.matmul(la.sqrtm(observation_noise),
                                        np.random.randn(self.dims_observations,self._num_samples))
        
        if plot: self.plot_simulated_data()
        return self.noisy_data if noisy else self.data
            
    def integrate_model(self):
        true_state = np.zeros((len(self.initial_conditions),self._num_samples)) # allocate
        true_state[:,0] = self.initial_conditions
        for n in range(self._num_samples-1):
            x_temp = true_state[:,n]
            true_state[:,n+1] = self.integrate(state=x_temp, params=self.parameters[:,n])
        return true_state

    def plot_simulated_data(self):
        '''Plot simulation'''
        plt.rc('text', usetex=True)
        plt.figure(figsize=(10,2))
        plt.plot(self.noisy_data[0,:],'bd',markeredgecolor='blue', mfc='blue',ms=3,label='noisy data');
        plt.plot(self.observation_function(self.augmented_state).T,'k',linewidth=2,label='actual'); 
        plt.xlabel('t');
        plt.legend();
        plt.axis('tight')
        plt.title('Simulation')

def set_num_samples(total_time,dt_sample):
    return round(total_time/dt_sample)
def set_steps_per_sample(dt_sample,dt_integrate):
    return round(dt_sample/dt_integrate)

class epileptor_model(Model):
    def __init__(self,params=param_epileptor):
        '''x0 is tracked parameter'''
        self.x0, self.y0 = params['x0'], params['y0']
        self.tau0, self.tau1, self.tau2 = params['tau0'], params['tau1'], params['tau2']
        self.Irest1, self.Irest2 = params['Irest1'], params['Irest2']
        self.gamma = params['gamma']
        self.initial_conditions = [params['x1_init'],params['y1_init'],
                                   params['z_init'], params['x2_init'],
                                   params['y2_init'],params['g_init']]
        self.noise = [params['noise_ensemble1'], 0.0, 0., params['noise_ensemble2'], 0., 0.]

        self.integrator = 'ruku4'

        self.var_names = ['x1','y1','z','x2','y2','g']
        self.parameter_names = ['x0']
        
        self.dims_params = len(self.var_names)
        self.dims_state_vars = len(self.parameter_names)
        self.dims_observations = 1
        self.dims_augmented_state = self.dims_params + self.dims_state_vars
                
        self._total_time = 10 # 2500 for epileptor, 160 for FN
        self._dt_sample = 0.1
        self._dt_integrate = self._dt_sample
        self._num_samples = int(set_num_samples(self._total_time,self._dt_sample))
        self._steps_per_sample = int(set_steps_per_sample(self._dt_sample,self._dt_integrate))

        x0_parameter = (self.x0*np.ones(self._num_samples)).reshape(1,self._num_samples)
        self.parameters = x0_parameter
        
    
    def _get_total_time(self):
        return self._total_time
    def _set_total_time(self, value):
        self._total_time = value
        self._num_samples = set_num_samples(self._total_time,self._dt_sample)
    def _get_dt_sample(self):
        return self._dt_sample
    def _set_dt_sample(self,value):
        self._dt_sample = value
        self._steps_per_sample = set_steps_per_sample(self._dt_sample,self._dt_integrate)
        self._num_samples = set_num_samples(self._total_time,self._dt_sample)
    def _get_dt_integrate(self):
        return self._dt_integrate
    def _set_dt_integrate(self,value):
        self._dt_integrate = value
        self._steps_per_sample = set_steps_per_sample(self._dt_sample,self._dt_integrate)
    
    total_time = property(_get_total_time,_set_total_time)
    dt_sample = property(_get_dt_sample,_set_dt_sample)
    dt_integrate = property(_get_dt_integrate,_set_dt_integrate)
    num_samples = property(lambda self: self._num_samples)
    steps_per_sample = property(lambda self: self._steps_per_sample)
    
    def integrate(self,state,params):
        switcher = {
            'ruku4': integrators.ruku4,
            'euler': integrators.euler,
            'euler_maruyama': integrators.euler_maruyama,
            'test_integrator': integrators.test_integrator
        }
        return switcher[self.integrator](self.model_function,state,params,
                                         self.dt_integrate,self.steps_per_sample,self.noise)

    def model_function(self,state,parameters):
        x1, y1, z, x2, y2, g = state
        x0 = parameters.reshape(x1.shape)
        x1_dot = y1 - self.f1(x1,x2,z) - z + self.Irest1
        y1_dot = self.y0 - 5.*x1*x1 - y1
        z_dot = 1/self.tau0*(4*(x1 - x0) - z)
        x2_dot = -y2 + x2 - x2**3 + self.Irest2 + 2.*g - 0.3*(z - 3.5)
        y2_dot = 1/self.tau2*(-y2 + self.f2(x2))
        g_dot = -self.gamma*(g - 0.1*x1)
        return np.array([x1_dot, y1_dot, z_dot, x2_dot, y2_dot, g_dot])
    
    def observation_function(self,augmented_state):
        x1 = augmented_state[self.dims_params,:]
        x2 = augmented_state[self.dims_params+3,:]
        return -x1 + x2
    
    def set_initial_estimate(self,initial_estimate):
        x1 = self.noisy_data[0,0]/2.
        x2 = self.noisy_data[0,0]/2. + x1
        initial_estimate[self.dims_params] = x1
        initial_estimate[self.dims_params+3] = x2
        return initial_estimate
    
    def transition_function(self, augmented_state):
        parameters, state = np.split(augmented_state,[self.dims_params,])
        state = ruku4(self.model_function,state,parameters,self.dt_integrate,
                      self.steps_per_sample,self.noise)
        return np.vstack((parameters,state))
    
    def f1(self,x1,x2,z):
        return ( x1**3 - 3*x1**2 ) * (x1 < 0) + ( x1*(x2 - 0.6*(z-4)**2) ) * (x1 >=0)
    
    def f2(self,x2):
        return 0 * (x2 < -0.25) + ( 6*(x2 + 0.25) ) * (x2 >= -0.25)


def load_protocols():
    """Load simulation using params from Jirsa, 2014."""
    protocols = [Protocol(prot_id='default')]
    target = [epileptor_model().generate_simulation(noisy=True)]

    return protocols, target

def rmse(estimate, target):
    logging_debug('estimate: %f, %s',estimate,type(estimate))
    logging_debug('target: %f, %s',target,type(target))
    return np.sqrt(((estimate - target)**2).mean())

def protocol_outcome(protocol, param=param_epileptor):
    """Compute the average synaptic gain for a given stimulation protocol and
    model parameters.

    :param protocol: epileptor_util.Protocol
        The stimulation protocol.
    :param model: dict
        Parameters of the Epileptor model
    """
    
    estimate = epileptor_model(param).generate_simulation() # Fix this later to actually run different protocols and parameters
    
    return estimate
