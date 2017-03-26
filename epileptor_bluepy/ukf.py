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
            value.append(1e-16)
        return value[:dim]


def keep_in_bounds(value, bounds=(-100, 100)):
    type_temp = type(value)
    value = np.array(value)
    value[value < bounds[0]] = bounds[0]
    value[value > bounds[1]] = bounds[1]
    return type_temp(value) if type_temp != np.ndarray else value


class unscented_kalman_filter(Model):
    def __init__(self, model, parameter_sigma=15e-3, state_sigma=15e-3,
                 observation_sigma=None):
        self.model = model
        self.parameter_sigma = parameter_sigma
        self.state_sigma = state_sigma
        if observation_sigma is None:
            self.observation_sigma = np.std(model.noisy_data)
        else:
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
        self.innovation = np.zeros((model.dims_observations,
                                    model.num_samples))

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
        for i in range(model.dims_params):
            sigmas = self.Q
            if sigmas[i] < 1e-15:
                self.estimated_state[i, 0] = model.augmented_state[i, 0]

    def set_covariances(self):
        model = self.model
        self.Pxx = np.zeros((model.dims_augmented_state,
                             model.dims_augmented_state, model.num_samples))
        self.Q = keep_in_bounds(create_sigma_list(
            self.parameter_sigma, model.dims_params),
            bounds=(1e-16, 100))
        self.R = keep_in_bounds(create_sigma_list(
            self.state_sigma, model.dims_state_vars),
            bounds=(1e-16, 100))
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
        self.sigma_points = keep_in_bounds(sigma_points, bounds=(-100, 100))

    def unscented_kalman(self, initial_estimate=[]):
        model = self.model
        self.set_initial_estimate(initial_estimate=initial_estimate)
        for k in range(1, model.num_samples):
            self.k = k
            try:
                self.voss_unscented_transform(k)
            except la.LinAlgError:
                logging_debug('LinAlgError\nk: {}\nPxx: {}'.format(
                    k, self.Pxx[:, :, k - 1]))
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
        Y = model.observation_function(X).\
            reshape(model.dims_observations, -1)

        Pxx = symmetrize(covariance(X, X))
        Pyy = covariance(Y, Y) + self.observation_sigma
        Pxy = covariance(X, Y)

        # K = keep_in_bounds(np.matmul(Pxy, la.inv(Pyy)),
        #                    bounds=(-1., 1.))
        K = np.matmul(Pxy, la.inv(Pyy))
        innovation = model.noisy_data[:, k] - np.mean(Y, 1)
        xhat = np.mean(X, 1) + np.matmul(K, innovation)

        Pxx = symmetrize(Pxx - np.matmul(K, Pxy.T))

        self.estimated_state[:, k] = keep_in_bounds(xhat)
        self.Pxx[:, :, k] = keep_in_bounds(Pxx, bounds=(-100, 100))
        self.Ks[:, :, k] = K
        self.innovation[:, k] = innovation

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
            plt.plot(model.time,
                     model.augmented_state[model.dims_params + i, :],
                     lw=2, label=model.var_names[i])
            plt.plot(model.time,
                     self.estimated_state[model.dims_params + i, :],
                     'r--', lw=2)
            if separated:
                plt.title(model.var_names[i])
                plt.xlabel('t')
                plt.axis('tight')
                plt.show()
        if not separated:
            plt.title('Estimated State Variables')
            plt.legend()
            plt.xlabel('t/dt')
            plt.axis('tight')
            plt.show()
        plt.figure(figsize=(10, 2))
        plt.plot(model.time, model.noisy_data[0], 'k')
        plt.plot(model.time,
                 model.observation_function(self.estimated_state), 'r')
        plt.title('Observed Results')
        plt.axis('tight')
        plt.xlabel('t')
        plt.show()

        for i in range(model.dims_params):
            if self.Q[i] < 1e-15:
                continue
            if i == 0 or (separated and i > 0):
                plt.figure(figsize=(10, 2))
            plt.plot(model.time, model.augmented_state[i, :], 'k',
                     linewidth=2, label=model.parameter_names[i])
            plt.plot(model.time,
                     self.estimated_state[i, :], 'm', linewidth=2)
            plt.plot(model.time,
                     self.estimated_state[i, :] + self.errors[i, :], 'm')
            plt.plot(model.time,
                     self.estimated_state[i, :] - self.errors[i, :], 'm')
            if separated:
                plt.title(model.parameter_names[i])
                plt.xlabel('t')
                plt.axis('tight')
                plt.show()
        if not separated:
            plt.title('Estimated Parameters')
            plt.legend()
            plt.xlabel('t')
            plt.axis('tight')
            plt.show()
