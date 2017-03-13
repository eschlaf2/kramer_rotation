import logging
import numpy as np
import integrators
import matplotlib.pyplot as plt
from integrators import ruku4
import scipy.linalg as la
from epileptor_util import Model


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


def covariance(X, Y):
    '''Calculates biased covariance (or cross covariance)
    of array-like X and Y'''
    num_samples = len(X[0, :])
    X_centered = mean_center(np.array(X))
    Y_centered = mean_center(np.array(Y))
    return np.matmul(X_centered, Y_centered.T / num_samples)


def mean_center(X):
    '''Centers the mean of rows of X (array-like) around 0.'''
    return np.array([x - np.mean(x) for x in X])


def symmetrize(A):
    '''Numerical safety'''
    return (A + np.transpose(A)) / 2


def create_sigma_list(value, dim):
    if type(value) in [float, int]:
        return list(value * np.ones(dim))
    else:
        value = list(value)
        while len(value) < dim:
            value.append(0.)
        return value[:dim]


def keep_in_bounds(value, bounds=(-30, 30)):
    type_temp = type(value)
    value = np.array(value)
    value[value < bounds[0]] = bounds[0]
    value[value > bounds[1]] = bounds[1]
    return type_temp(value) if type_temp != np.ndarray else value


class unscented_kalman_filter(Model):
    def __init__(self, model, process_sigma=15e-3, observation_sigma=15e-3):
        self.model = model
        self.process_sigma = process_sigma
        self.observation_sigma = observation_sigma
        self.sigma_points = []
        self.results = []
        self.initial_estimate = []
        self.set_covariances()
        self.estimated_state = np.zeros((model.dims_augmented_state,
                                         model.num_samples))
        self.errors = np.zeros((model.dims_augmented_state, model.num_samples))
        self.Ks = np.zeros((model.dims_augmented_state,
                            model.dims_observations, model.num_samples))

    def set_initial_estimate(self, initial_estimate=None, randrange=0.):
        model = self.model
        if initial_estimate is None:
            initial_estimate = 'model'
        switcher = {
            'model': model.set_initial_estimate(
                np.zeros(model.dims_augmented_state)),
            'zeros': np.zeros(model.dims_augmented_state),
            'exact': model.augmented_state[:, 0]
        }
        if type(initial_estimate) == str:
            self.estimated_state[:, 0] = switcher[initial_estimate]
        else:
            self.estimated_state[:, 0] = \
                create_sigma_list(initial_estimate, model.dims_augmented_state)
        if randrange > 0:
            random_smear = np.random.rand(model.dims_augmented_state) * \
                randrange * 2 - randrange
            self.estimated_state[:, 0] += random_smear

        # if initial_estimate:
        #     self.estimated_state[:, 0] = initial_estimate
        # elif randrange and np.all(np.array(randrange) >= 0):
        #     if len(randrange) == 1:
        #         r0, rf = 0, int(np.fix(randrange))
        #     else:
        #         r0, rf = int(np.fix(randrange[0])), int(np.fix(randrange[1]))
        #     r = 2 * rf
        #     initial_estimate = -rf + np.random.randint(
        #         r0, r, model.augmented_state[:, 0].shape) + \
        #         model.augmented_state[:, 0]
        #     self.estimated_state[:, 0] = \
        #         model.set_initial_estimate(initial_estimate)
        # elif self.initial_estimate:
        #     self.estimated_state[:, 0] = self.initial_estimate
        # else:
        #     self.estimated_state[:, 0] = model.augmented_state[:, 0]

    def set_covariances(self):
        model = self.model
        self.Pxx = np.zeros((model.dims_augmented_state,
                             model.dims_augmented_state, model.num_samples))
        # sigmas = {}
        # sigmas['Q'] = self.process_sigma
        # sigmas['R'] = self.observation_sigma
        # Pxx_init = []
        # for n in model.Pxx0:
        #     Pxx_init = np.hstack((Pxx_init, sigmas[n]))
        # Pxx_init = np.diag(Pxx_init)
        # self.Q = np.diag(sigmas['Q'])
        # self.R = np.diag(sigmas['R'])
        self.Q = create_sigma_list(
            self.process_sigma, model.dims_params)
        self.R = create_sigma_list(
            self.observation_sigma, model.dims_state_vars)
        # self.Pxx[:, :, 0] = Pxx_init
        self.Pxx[:, :, 0] = np.diag(np.hstack((self.Q, self.R)))

    def generate_sigma_points(self, k):
        '''Why have the extra terms from the Cholesky decomp??
        Why not just use sigma in each direction?'''
#         dims = len(self.estimated_state[:,k])
        xhat = self.estimated_state[:, k - 1]
        Pxx = self.Pxx[:, :, k - 1]
        num_sigma_points = 2 * self.model.dims_augmented_state
        Pxx = symmetrize(Pxx)
        xsigma = la.cholesky(self.model.dims_augmented_state * Pxx, lower=True)
        sigma_points = np.hstack((xsigma, -xsigma))
        for i in range(num_sigma_points):
            sigma_points[:, i] += xhat
        self.sigma_points = sigma_points

    def unscented_kalman(self, initial_estimate=[]):
        model = self.model
        self.set_initial_estimate(initial_estimate=initial_estimate)
        for k in range(1, model.num_samples):
            try:
                self.voss_unscented_transform(k)
            except la.LinAlgError:
                self.Pxx[:, :, k - 1] = self.Pxx[:, :, k - 2]
                self.voss_unscented_transform(k)
            if model.dims_params > 0:
                self.Pxx[:model.dims_params, :model.dims_params, k] = \
                    np.diag(self.Q)
            self.errors[:, k] = np.sqrt(np.diag(self.Pxx[:, :, k]))

    def voss_unscented_transform(self, k):
        model = self.model
        Pxx = symmetrize(self.Pxx[:, :, k - 1])
        self.generate_sigma_points(k)
        sigma_points = self.sigma_points

        X = model.transition_function(sigma_points)
        Y = model.observation_function(X).reshape(1, -1)

        Pxx = symmetrize(covariance(X, X))
        Pyy = covariance(Y, Y) + self.observation_sigma
        Pxy = covariance(X, Y)

        K = np.matmul(Pxy, la.inv(Pyy))
        xhat = np.mean(X, 1) + \
            np.matmul(K, (model.noisy_data[:, k] - np.mean(Y, 1)))

        Pxx = symmetrize(Pxx - np.matmul(K, Pxy.T))

        self.estimated_state[:, k] = keep_in_bounds(xhat)
        self.Pxx[:, :, k] = keep_in_bounds(Pxx, bounds=(-100, 100))
        self.Ks[:, :, k] = K

    def print_results(self):
        '''Prints results of Kalman filtering.'''
        model = self.model
        results = {}
        results['chisq'] = \
            np.mean(sum((model.augmented_state - self.estimated_state)**2))
        results['est'] = self.estimated_state[:model.dims_params, -1]
        results['error'] = self.errors[:model.dims_params, -1]
        results['meanest'] = \
            np.mean(self.estimated_state[:model.dims_params, :], 1)
        results['meanerror'] = np.mean(self.errors[:model.dims_params, :], 1)
        for key, value in results.items():
            print('{0:15s}{1}'.format(key + ':', str(value)))
        self.results = results

    def plot_filter_results(self, separated=False):
        '''Plots results of Kalman filtering'''
        model = self.model
        plt.rc('text', usetex=True)
        plt.figure(figsize=(10, 2))
        for i in range(model.dims_state_vars):
            if separated and i > 0:
                plt.figure(figsize=(10, 2))
            plt.plot(model.augmented_state[model.dims_params + i, :],
                     lw=2, label=model.var_names[i])
            plt.plot(self.estimated_state[model.dims_params + i, :],
                     'r--', lw=2)
            if separated:
                plt.title(model.var_names[i])
                plt.xlabel('t/dt')
                plt.axis('tight')
                plt.show()
        if not separated:
            plt.title('Estimated State Variables')
            plt.legend()
            plt.xlabel('t/dt')
            plt.axis('tight')
            plt.show()
        plt.figure(figsize=(10, 2))
        plt.plot(model.noisy_data[0], 'k')
        plt.plot(model.observation_function(self.estimated_state), 'r')
        plt.title('Observed Results')
        plt.axis('tight')
        plt.xlabel('t/dt')
        plt.show()

        plt.figure(figsize=(10, 2))
        for i in range(model.dims_params):
            if separated and i > 0:
                plt.figure(figsize=(10, 2))
            plt.plot(model.augmented_state[i, :], 'k',
                     linewidth=2, label=model.parameter_names[i])
            plt.plot(self.estimated_state[i, :], 'm', linewidth=2)
            plt.plot(self.estimated_state[i, :] + self.errors[i, :], 'm')
            plt.plot(self.estimated_state[i, :] - self.errors[i, :], 'm')
            if separated:
                plt.title(model.parameter_names[i])
                plt.xlabel('t/dt')
                plt.axis('tight')
                plt.show()
        if not separated:
            plt.title('Estimated Parameters')
            plt.legend()
            plt.xlabel('t/dt')
            plt.axis('tight')
            plt.show()
