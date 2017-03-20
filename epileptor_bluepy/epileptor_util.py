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
import matplotlib.pyplot as plt
from integrators import ruku4
from pyedflib import EdfReader
import copy

logging.basicConfig(level=logging.WARN)

# Note: having debug logging statements increases the run time by ~ 25%,
# because they exist in tight loops, and expand their outputs, even when
# debug is off, so we disable logging if possible.  Set this to true if
# verbose output is needed
LOGGING_DEBUG = True

plot = True


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
    'observation_sigmas': None,
    'noise_ensemble1': 25e-3,
    'noise_ensemble2': 25e-2,
    'a': 5.,
    'b': 4.,
    'c': 0.3,
    'd': 3.5}


class Protocol(object):

    """Protocol"""

    def __init__(self, params=None,
                 total_time=2500, prot_id=None, **kwargs):
        if not params:
            params = dict(param_epileptor)
        self.prot_id = prot_id
        self.params = dict(params)
        self.total_time = total_time
        for key, value in kwargs.iteritems():
            self.params[key] = float(value)


class Model(object):
    def __init__(self):
        self.augmented_state = []

    def generate_simulation(self, plot=plot):
        '''
            Simulates true and noisy trajectory based on previously
            defined model and parameter functions
            (Uses global vars)
            '''

        # Simulate model
        true_state = self.integrate_model()
        self.augmented_state = np.vstack((self.parameters * np.ones(
            (self.dims_params, self._num_samples)), true_state))\
            if self.dims_params > 0 \
            else true_state
        self.dims_augmented_state = self.dims_params + self.dims_state_vars

        # Observation noise
        if self.observation_sigmas is None:
            self.observation_sigmas = list(np.sqrt([0.2 * 0.2 * np.var(
                self.observation_function(self.augmented_state))]))
        if type(self.observation_sigmas) != list:
            self.observation_sigmas = [self.observation_sigmas]
        if len(self.observation_sigmas) > self.dims_observations:
            self.observation_sigmas = \
                self.observation_sigmas[:self.dims_observations]
        while len(self.observation_sigmas) < self.dims_observations:
            self.observation_sigmas.append(0.)

        observation_noise = np.diag(self.observation_sigmas)

        # Create noisy data from true trajectory
        self.data = self.observation_function(self.augmented_state)
        self.noisy_data = self.data + \
            np.matmul(observation_noise,
                      np.random.randn(self.dims_observations,
                                      self._num_samples))

        if plot:
            self.plot_simulated_data()
        return self.noisy_data  # if noisy else self.data

    def integrate(self, state, time_varying_params):
        switcher = {
            'ruku4': integrators.ruku4,
            'euler': integrators.euler,
            'euler_maruyama': integrators.euler_maruyama,
            'test_integrator': integrators.test_integrator
        }
        return switcher[self.integrator](self.model_function,
                                         state,
                                         time_varying_params,
                                         self.dt_integrate,
                                         self.steps_per_sample,
                                         self.noise)

    def integrate_model(self):
        true_state = np.zeros((len(self.initial_conditions),
                               self._num_samples))  # allocate
        true_state[:, 0] = self.initial_conditions
        for n in range(self._num_samples - 1):
            x_temp = true_state[:, n]
            true_state[:, n + 1] = \
                self.integrate(state=x_temp,
                               time_varying_params=self.parameters[:, n])
            if any(np.isinf(true_state[:, n + 1])):
                break
        return true_state

    def plot_simulated_data(self):
        '''Plot simulation'''
        plt.rc('text', usetex=True)
        plt.figure(figsize=(10, 2))
        plt.plot(self.noisy_data[0, :],
                 'bd', markeredgecolor='blue',
                 mfc='blue', ms=3, label='noisy data')
        plt.plot(self.observation_function(self.augmented_state).T,
                 'k', linewidth=2, label='actual')
        # plt.figure()
        # for i in range(self.dims_state_vars):
        # plt.plot(self.augmented_state[self.dims_params + 2, :],
        #          label=self.var_names[2])
        plt.xlabel('t')
        plt.legend()
        plt.axis('tight')
        plt.title('Simulation')


def set_num_samples(total_time, dt_sample):
    return round(total_time / dt_sample)


def set_steps_per_sample(dt_sample, dt_integrate):
    return round(dt_sample / dt_integrate)


class epileptor_model(Model):
    def __init__(self, params=None,
                 total_time=2500, dt_sample=0.1,
                 **kwargs):
        '''x0 is tracked parameter'''
        if not params:
            params = {}
            for key, value in param_epileptor.iteritems():
                params[key] = value
        for key, value in kwargs.iteritems():
            params[key] = float(value)
        self.params = params
        self.x0, self.y0 = params['x0'], params['y0']
        self.tau0, self.tau1, self.tau2 = \
            params['tau0'], params['tau1'], params['tau2']
        self.Irest1, self.Irest2 = params['Irest1'], params['Irest2']
        self.gamma = params['gamma']
        self.initial_conditions = [params['x1_init'], params['y1_init'],
                                   params['z_init'], params['x2_init'],
                                   params['y2_init'], params['g_init']]
        self.noise = [params['noise_ensemble1'],
                      0.0,
                      0.,
                      params['noise_ensemble2'],
                      0.,
                      0.]
        self.observation_sigmas = params['observation_sigmas']

        self.a, self.b = params['a'], params['b']
        self.c, self.d = params['c'], params['d']

        self.integrator = 'ruku4'

        self.var_names = ['x1', 'y1', 'z', 'x2', 'y2', 'g']
        self.parameter_names = ['$I_{ext1}$', '$I_{ext2}$', '$I_{extz}$']

        self.dims_params = len(self.parameter_names)
        self.dims_state_vars = len(self.var_names)
        self.dims_observations = 1
        self.dims_augmented_state = self.dims_params + self.dims_state_vars

        self._total_time = total_time  # 2500 for epileptor, 160 for FN
        self._dt_sample = dt_sample
        self._dt_integrate = self._dt_sample
        self._num_samples = int(set_num_samples(
            self._total_time, self._dt_sample))
        self._steps_per_sample = int(set_steps_per_sample(
            self._dt_sample, self._dt_integrate))

        I_ext1 = (
            self.Irest1 * np.ones(self._num_samples)).reshape(
            1, self._num_samples)
        I_ext2 = (
            self.Irest2 * np.ones(self._num_samples)).reshape(
            1, self._num_samples)
        I_extz = (
            0. * np.ones(self._num_samples)).reshape(
            1, self._num_samples)
        self.parameters = np.vstack((I_ext1, I_ext2, I_extz))

    def _get_total_time(self):
        return self._total_time

    def _set_total_time(self, value):
        self._total_time = value
        self._num_samples = set_num_samples(
            self._total_time, self._dt_sample)

    def _get_dt_sample(self):
        return self._dt_sample

    def _set_dt_sample(self, value):
        self._dt_sample = value
        self._steps_per_sample = \
            set_steps_per_sample(self._dt_sample, self._dt_integrate)
        self._num_samples = \
            set_num_samples(self._total_time, self._dt_sample)

    def _get_dt_integrate(self):
        return self._dt_integrate

    def _set_dt_integrate(self, value):
        self._dt_integrate = value
        self._steps_per_sample = \
            set_steps_per_sample(self._dt_sample, self._dt_integrate)

    total_time = property(_get_total_time, _set_total_time)
    dt_sample = property(_get_dt_sample, _set_dt_sample)
    dt_integrate = property(_get_dt_integrate, _set_dt_integrate)
    num_samples = property(lambda self: self._num_samples)
    steps_per_sample = property(lambda self: self._steps_per_sample)

    def model_function(self, state, time_varying_params):
        '''
        TODO: unscented kalman filter parameter
        '''
        x1, y1, z, x2, y2, g = state
        # I_ext = time_varying_params.reshape(x1.shape)
        I_ext1, I_ext2, I_extz = time_varying_params
        x1_dot = y1 - self.f1(x1, x2, z) - z + I_ext1  # self.Irest1
        y1_dot = self.y0 - self.a * x1 * x1 - y1  # a = 5., tvb param d
        z_dot = 1. / self.tau0 * \
            (self.b * (x1 - self.x0) - z) + I_extz  # b = 4., tvb const
        x2_dot = -y2 + x2 - x2**3 + I_ext2 + \
            2. * g - self.c * (z - self.d)  # + self.Irest2 c = 0.3, d = 3.5
        y2_dot = (-y2 + self.f2(x2)) / self.tau2
        g_dot = -self.gamma * (g - 0.1 * x1)
        return np.array([x1_dot, y1_dot, z_dot, x2_dot, y2_dot, g_dot])

    def observation_function(self, augmented_state):
        x1 = augmented_state[self.dims_params, :]
        x2 = augmented_state[self.dims_params + 3, :]
        return -x1 + x2

    def set_initial_estimate(self, initial_estimate):
        x1 = self.noisy_data[0, 0] / 2.
        x2 = self.noisy_data[0, 0] / 2. + x1
        initial_estimate[self.dims_params] = x1
        initial_estimate[self.dims_params + 3] = x2
        return initial_estimate

    def transition_function(self, augmented_state):
        parameters, state = np.split(augmented_state, [self.dims_params, ])
        state = ruku4(self.model_function,
                      state,
                      parameters,
                      self.dt_integrate,
                      self.steps_per_sample,
                      self.noise)
        return np.vstack((parameters, state))

    def f1(self, x1, x2, z):
        return (x1**3 - 3 * x1**2) * (x1 < 0) + \
            (x1 * (x2 - 0.6 * (z - 4)**2)) * (x1 >= 0)

    def f2(self, x2):
        return 0. * (x2 < -0.25) + (6 * (x2 + 0.25)) * (x2 >= -0.25)


def load_protocols(plot=plot, total_time=[], dt_sample=0.1):
    """Load simulation using params from Jirsa, 2014."""

    if not total_time:
        total_time = 2500
    # protocols = [Protocol(prot_id='default'),
    #              Protocol(prot_id='clean', observation_sigmas=0., tau0=1000),
    #              Protocol(prot_id='noiseless',
    #                       observation_sigmas=0.,
    #                       noise_ensemble1=0.,
    #                       noise_ensemble2=0.,
    #                       tau0=2000)]
    # target = [epileptor_model(params=protocols[0].params,
    #                           total_time=total_time, dt_sample=dt_sample).
    #           generate_simulation(plot=plot),
    #           epileptor_model(params=protocols[1].params,
    #                           total_time=total_time, dt_sample=dt_sample).
    #           generate_simulation(plot=plot),
    #           epileptor_model(params=protocols[2].params,
    #                           total_time=total_time, dt_sample=dt_sample).
    #           generate_simulation(plot=plot)]
    protocols = [Protocol(prot_id='default', total_time=total_time)]
    target = [epileptor_model(params=protocols[0].params,
                              total_time=total_time, dt_sample=dt_sample).
              generate_simulation(plot=plot)]

    # f = EdfReader(
    #     '/Users/emilyschlafly/BU/Kramer_rotation/ieeg_data/' +
    #     'I002_A0003_D010/outputEdf_EDF/outputEdf_0.edf')
    # chan = 0
    # data = f.readSignal(chan)
    # sample_freq = f.getSampleFrequency(chan)
    # if not total_time:
    #     num_samples = len(data)
    #     total_time = num_samples / sample_freq
    # data = data[::int(round(sample_freq * dt_sample))]
    # data = data[:int(total_time / dt_sample)] / 1000.
    # total_time = min(total_time, len(data) * dt_sample)

    # f._close()
    # del f

    # if plot:
    #     plt.plot(np.arange(0, total_time, dt_sample), data)

    # target = [data]
    # protocols = [Protocol(prot_id='1', total_time=total_time)]

    return protocols, target


def rmse(estimate, target):
    logging_debug('estimate: %f, %s', estimate, type(estimate))
    logging_debug('target: %f, %s', target, type(target))
    return np.sqrt(((estimate - target)**2).mean())


def protocol_outcome(protocol, param=param_epileptor, plot=plot):
    """Compute the average synaptic gain for a given stimulation protocol and
    model parameters.

    :param protocol: epileptor_util.Protocol
        The stimulation protocol.
    :param model: dict
        Parameters of the Epileptor model
    """

    estimate = epileptor_model(param, total_time=protocol.total_time).\
        generate_simulation(plot=plot)

    return estimate
