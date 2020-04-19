from experiment import Experiment
import skopt

from preprosses import preprocess

import matplotlib.pyplot as plt



HPO_PARAMS = {'n_calls':5,
              'n_random_starts':1,
              'base_estimator':'ET',
              'acq_func':'EI',
              'xi':0.02,
              'kappa':1.96,
              'n_points':10000,
             }


SPACE =[skopt.space.Categorical(['average', 'max','min'], name='valarou'),
        skopt.space.Categorical(['average', 'max','min'], name='app'),
        skopt.space.Categorical(['average', 'max','min'], name='time'),
        skopt.space.Categorical(['average', 'max','min'], name='month'),]
        # skopt.space.Categorical([True,False], name='transform_targets'),]

# options.update(mod_opt_xgb)
# exp = Experiment(options)
# res = exp.train_and_test()
# print(res)
@skopt.utils.use_named_args(SPACE)
def search(**params):
        global exp
        all_params = params
        # global data 
        transf_opts = ['average' , all_params['valarou'] ,all_params['valarou'], all_params['app'], all_params['app'], all_params['app'], all_params['app'],
        all_params['app'], all_params['app'], all_params['app'], all_params['app'], all_params['app'], all_params['app'], all_params['app'], all_params['app'], all_params['app'], all_params['app'], all_params['app'] , all_params['time'], all_params['time'], all_params['time'], all_params['time'], all_params['month'],
        all_params['month'], all_params['month'], all_params['month'], all_params['month']]

        mod_opt_xgb = { 'max_depth'          : 15, 
                        'aplha'              : 200,
                        'colsample_bytree'   : 0.03,
                        'n_estimators'       : 10000,
                        'lr'                 : 0.01,
                        'use_pca'            : True,
                        'pca_var_hold'       : .995,
                        'transform_targets'  : True,
                        'max_delta_step'     : 2000, 
                        'gamma'              : 0.07 

        }
        print('Experiment with vars:  ',all_params['valarou'],all_params['app'],all_params['time'],all_params['month'])
        # print(mod_opt_xgb)
        options = {     'model_type'        : 'xgb', # xgb  mlp
                        'win_size'          : 3,
                        'batch_size'        : 128,
                        'data'              : None,
                        'transform_targets' : True,
                        'split'             : 0.2,
                        'model_options' : mod_opt_xgb ,
                        'methods'       : transf_opts
        }
        exp = Experiment(options)

        # exp.update_opt(options)
        res = exp.train_and_test()
        print('accuracy: ',res)
        return 1 - res

from skopt.plots import plot_convergence

dummy_results = skopt.dummy_minimize(search, SPACE,n_calls=5)
skopt.dump(dummy_results, 'results/dummy_transf_hyper.pkl')
plt.figure(1)
plot_convergence(dummy_results)
plt.savefig("plots/dummy_transf_hyper_accinv.png")

gp_results = skopt.gp_minimize(search, SPACE,n_calls=5)
skopt.dump(gp_results, 'results/gp_transf_hyper.pkl')
plt.figure(2)
plot_convergence(gp_results)
plt.savefig("plots/gp_transf_hyper_accinv.png")

gbrt_results = skopt.gbrt_minimize(search, SPACE, **HPO_PARAMS)
skopt.dump(gbrt_results, 'results/gbrt_transf_hyper.pkl')
plt.figure(3)
plot_convergence(gbrt_results)
plt.savefig("plots/gbrt_transf_hyper_accinv.png")
# print(results)



# plot_convergence(results)
# plt.savefig("plots/xgb_hyper_accinv.png")
results = [('random_results', dummy_results),
           ('gbrt_results', gbrt_results),
           ('gp_results', gp_results)]
plt.figure(4)
plot_convergence(*results)
plt.savefig("plots/all_transf_hyper_accinv.png")
