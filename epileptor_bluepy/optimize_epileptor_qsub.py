import bluepyopt as bpop
import epileptor_evaluator
import pickle


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    hall_of_fame = []
    for i in range(1):
        evaluator = epileptor_evaluator.\
            Epileptor_Evaluator(plot=False, total_time=2500, dt_sample=0.1)
        opt = bpop.optimisations.\
            DEAPOptimisation(evaluator, offspring_size=50,
                             eta=20, mutpb=0.3, cxpb=0.7,
                             seed=i, use_scoop=True)
        final_pop, hof_temp, log, hst = opt.run(max_ngen=100)
        hall_of_fame.append(hof_temp[0])
        params = evaluator.get_param_dict(hof_temp[0])
        save_obj(params, 'params{}'.format(i))
    save_obj(hall_of_fame, 'hall_of_fame')
