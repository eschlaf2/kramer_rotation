"""Main Graupner-Brunel STDP example script"""

import numpy

import bluepyopt as bpop
import epileptor_util
import logging

logging.basicConfig(level=logging.WARN)

LOGGING_DEBUG = False

def logging_debug_vec(fmt, vec):
    '''log to debug a vector'''
    if LOGGING_DEBUG:
        logging.debug(fmt, ', '.join(map(str, vec)))


def logging_debug(*args):
    '''wrapper to log to debug a vector'''
    if LOGGING_DEBUG:
        logging.debug(*args)


def get_epileptor_params(params):
    """Create the fixed parameter set for Epileptor model.

    :param individual: iterable
    :rtype : dict
    """
    ep_param = {
    'y0': 1.,
    'tau0': 2857.,
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
    'noise_ensemble2': 25e-2} # Fixed params; x0 will be optimized
                   
                  
    for param_name, param_value in params:
       ep_param[param_name] = param_value

    return ep_param

class Epileptor_Evaluator(bpop.evaluators.Evaluator):

    """Epileptor Evaluator"""

    def __init__(self):
        """Constructor"""

        super(Epileptor_Evaluator, self).__init__()
        # Graupner-Brunel model parameters and boundaries,
        # from (Graupner and Brunel, 2012)
        self.ep_params = [('x0', -2., 0.)]

        self.params = [bpop.parameters.Parameter
                       (param_name, bounds=(min_bound, max_bound))
                       for param_name, min_bound, max_bound in self.
                       ep_params]

        self.param_names = [param.name for param in self.params]

        self.protocols, self.target = epileptor_util.load_protocols() # protocols and targets for each protocol

        self.objectives = [bpop.objectives.Objective(protocol.prot_id)
                           for protocol in self.protocols]

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

        outcome = [epileptor_util.protocol_outcome(protocol, param_dict)
                    for protocol in self.protocols]

        return outcome

    def evaluate_with_lists(self, param_values):
        """Evaluate individual

        :param param_values: iterable
            Parameters list
        """
        param_dict = self.get_param_dict(param_values)

        err = []
        logging_debug_vec('evaluate_with_lists -> target:',self.target)
        for protocol, target in \
                zip(self.protocols, self.target):
            result = epileptor_util.protocol_outcome(protocol, param_dict)
            logging_debug_vec('result:%f',result)
            err.append(epileptor_util.rmse(target,result))

        return err
