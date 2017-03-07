"""Main Graupner-Brunel STDP example script"""

import bluepyopt as bpop
import epileptor_util
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.WARN)

LOGGING_DEBUG = False

plot = False  # Plot target traces


def logging_debug_vec(fmt, vec):
    '''log to debug a vector'''
    if LOGGING_DEBUG:
        # logging.debug(fmt, ', '.join(map(str, vec)))
        print(fmt, ', '.join(map(str, vec)))


def logging_debug(*args):
    '''wrapper to log to debug a vector'''
    if LOGGING_DEBUG:
        # logging.debug(*args)
        print(args)
        # for arg in args:
        #     print(arg)


def get_epileptor_params(params):
    """Create the fixed parameter set for Epileptor model.

    :param individual: iterable
    :rtype : dict
    """
    ep_param = {
        # 'a': 5.,
        # 'b': 4.,
        # 'c': 0.3,
        # 'd': 3.5,
        'x0': -1.6,
        'y0': 1.,
        # 'tau0': 2857.,
        'tau1': 1.0,
        'tau2': 10.,
        # 'Irest1': 3.1,
        # 'Irest2': 0.45,
        'gamma': 1e-2,
        'x1_init': 0.,
        # 'y1_init': -5.,
        # 'z_init': 3.,
        'x2_init': 0.,
        'y2_init': 0.,
        'g_init': 0.,
        'observation_sigmas': 0.,
        'noise_ensemble1': 0.,
        'noise_ensemble2': 0.}  # Fixed params;

    for param_name, param_value in params:
        ep_param[param_name] = param_value

    return ep_param


# class Epileptor_Evaluator(bpop.evaluators.Evaluator):
class Epileptor_Evaluator(object):

    """Epileptor Evaluator"""

    def __init__(self, plot=plot, total_time=2500, dt_sample=.1):
        """Constructor"""

        super(Epileptor_Evaluator, self).__init__()
        # Graupner-Brunel model parameters and boundaries,
        # from (Graupner and Brunel, 2012)
        self.ep_params = [('y1_init', -10., 0.),
                          ('z_init', 2., 6.),
                          ('tau0', 1000., 4000.),
                          ('a', 3., 5.5),
                          ('b', 2., 8.),
                          ('c', 0., 0.6),
                          ('d', 3., 4.5),
                          ('Irest1', 2.8, 4.),
                          ('Irest2', 0.2, 0.7)]
        # TODO: think about appropriate ranges for a and b

        self.params = [bpop.parameters.Parameter
                       (param_name, bounds=(min_bound, max_bound))
                       for param_name, min_bound, max_bound in self.
                       ep_params]

        self.param_names = [param.name for param in self.params]

        self.protocols, self.target = epileptor_util.\
            load_protocols(plot=plot, total_time=total_time,
                           dt_sample=dt_sample)
        # protocols and targets for each protocol

        self.objectives = [bpop.objectives.Objective(protocol.prot_id)
                           for protocol in self.protocols]

        self.plot = plot
        self.total_time = total_time

    def get_param_dict(self, param_values):
        """Build dictionary of parameters for the Epileptor model from an
        ordered list of values (i.e. an individual).

        :param param_values: iterable
            Parameters list
        """
        return get_epileptor_params(zip(self.param_names, param_values))

    def compute_protocol_outcome_with_lists(self, param_values):
        """Compute protocol outcome for all protocols.

        :param param_values: iterable
            Parameters list
        """
        param_dict = self.get_param_dict(param_values)

        outcome = [epileptor_util.
                   protocol_outcome(protocol, param_dict, plot=self.plot)
                   for protocol in self.protocols]

        return outcome

    def evaluate_with_lists(self, param_values):
        """Evaluate individual

        :param param_values: iterable
            Parameters list
        """
        param_dict = self.get_param_dict(param_values)

        err = []
        logging_debug_vec('evaluate_with_lists -> target:', self.target)
        for protocol, target in \
                zip(self.protocols, self.target):
            result = epileptor_util.protocol_outcome(protocol, param_dict,
                                                     plot=False)
            logging_debug_vec('evaluate_with_lists -> result:', result)
            logging_debug('evaluate_with_lists -> result type:', type(result))
            logging_debug_vec('evaluate_with_lists -> target:', target)
            logging_debug('evaluate_with_lists -> target type:', type(target))
            err.append(epileptor_util.rmse(target, result))
            if self.plot:
                fig, ax1 = plt.subplots(figsize=(10, 2))
                ax1.plot(target[0], 'r')
                # ax1.set_ylabel('target', color='r')
                # ax2 = ax1.twinx()
                ax1.plot(result[0], 'b')
                # ax2.set_ylabel('result', color='b')
                plt.axis('tight')
                plt.title('RMSE: {}'.format(err[-1]))
                plt.show()

        return err
