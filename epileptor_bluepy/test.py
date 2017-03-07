from __future__ import print_function
import bluepyopt as bpop
import epileptor_evaluator
import epileptor_util
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
# from scoop import futures

if __name__ == '__main__':
# if True:
    hof_scratch = []
    for i in range(1):
        evaluator = epileptor_evaluator.Epileptor_Evaluator(plot=False,
                                                            total_time=250,
                                                            dt_sample=0.1)
        opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=3,
                                                  eta=20, mutpb=0.3, cxpb=0.7,
                                                  seed=i,
                                                  use_scoop=True)
                                                  # map_function=futures.map)
        final_pop, hof_temp, log, hst = opt.run(max_ngen=3)
        hof_scratch.append(hof_temp[0])


# def helloWorld(value):
#     return "Hello World from Future #{0}".format(value)

# if __name__ == "__main__":
#     returnValues = list(futures.map(helloWorld, range(16)))
#     print("\n".join(returnValues))
