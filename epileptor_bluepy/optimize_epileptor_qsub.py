import bluepyopt as bpop
import epileptor_evaluator
from scipy import linalg as la
import matplotlib.pyplot as plt

if __name__ == '__main__':
	hall_of_fame = []
	for i in range(1):
        evaluator = epileptor_evaluator.Epileptor_Evaluator(plot=False,
                                                            total_time=250,
                                                            dt_sample=0.1)
        opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=3,
                                                  eta=20, mutpb=0.3, cxpb=0.7,
                                                  seed=i, use_scoop=True)
        final_pop, hof_temp, log, hst = opt.run(max_ngen=3)
        hall_of_fame.append(hof_temp[0])
np.save('test_run', hall_of_fame)
