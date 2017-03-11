import bluepyopt as bpop
import epileptor_evaluator
import pickle
import time
import sys


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    seed = sys.argv[1]
    total_time = 2500
    dt_sample = 0.1
    offspring_size = 3
    max_ngen = 3
    print(('\ntotal_time: {}\ndt_sample: {}\n' +
          'offspring_size: {}\nmax_ngen {}\n').
          format(total_time, dt_sample, offspring_size, max_ngen))
    hall_of_fame = []
    for s in sys.argv[1:]:
        t0 = time.time()
        evaluator = epileptor_evaluator.\
            Epileptor_Evaluator(plot=False, total_time=total_time, 
                                dt_sample=dt_sample)
        opt = bpop.optimisations.\
            DEAPOptimisation(evaluator, offspring_size=offspring_size,
                             eta=20, mutpb=0.3, cxpb=0.7,
                             seed=s, use_scoop=True)
        final_pop, hof_temp, log, hst = opt.run(max_ngen=max_ngen)
        t_total = time.time() - t0
        print('Time {}: {}'.format(s, t_total))
        hall_of_fame.append(hof_temp[0])
        params = evaluator.get_param_dict(hof_temp[0])
        save_obj(params, 'params{}'.format(s))
    save_obj(hall_of_fame, 'hall_of_fame')
