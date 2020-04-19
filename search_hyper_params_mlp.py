from experiment import Experiment
import skopt

from preprosses import preprocess

import matplotlib.pyplot as plt

HPO_PARAMS = {'n_calls':25,
              'n_random_starts':3,
              'base_estimator':'ET',
              'acq_func':'EI',
              'xi':0.2,
              'kappa':1.60,
              'n_points':10000,
              'n_jobs':2
             }
methods = ['average','max','max','max','max','max','max','max','max','max',
'max','max','max','max','max','max','max','max','average','average',
'average','average','average','average','average','average','average','average']

win_size = 3
batch_size =128

filename = 'data/RAW_Data.pickle'
preprocess_instance = preprocess(filename, window_size=win_size, methods=methods)
preprocess_instance.normalize()
preprocess_instance.bin(include_remainder=False)
processed_df = preprocess_instance.create_dataframe_pros()
indexNames = processed_df[ processed_df['user_id'] =='AS14.7' ].index
data= processed_df.drop(indexNames )
SPACE =[skopt.space.Real(0.0001, 0.1, name='lr', prior='log-uniform'),#
        # skopt.space.Real(0.994, 0.999, name='pca_var_hold'),#
        skopt.space.Categorical(['sgd', 'adam'], name='optim'),
        skopt.space.Categorical([32,64, 128,256], name='batch_size'),
        skopt.space.Categorical([True,False], name='use_pca'),]
        # skopt.space.Categorical([True,False], name='transform_targets'),]

options = {     'model_type'        : 'mlp', # xgb  mlp
                'win_size'          : win_size,
                'batch_size'        : 128,
                'data'              : data,
                'transform_targets' : True,
                'split'             : 0.2,
                'model_options' : None ,
                'methods'       : None
}
# options.update(mod_opt_xgb)
# exp = Experiment(options)
# res = exp.train_and_test()
# print(res)
exp = Experiment(options)
@skopt.utils.use_named_args(SPACE)
def search(**params):
        global exp
        all_params = params
        global data 
        mod_opt_mlp ={'exp_name'      : None, #default if dont want to specify 
              'win_size'      : win_size,
              'batch_size'    : all_params['batch_size'],
              'epochs'        : 25,
              'lr'            :all_params['lr'],
              'use_pca'       : all_params['use_pca'],
              'pca_var_hold'  : .995,
              'model_type'    : 'reg', #'cls'
              'transform_targets'  : True,
              'loss_fn'       : 'mse', #cross-entropy
              'optim'         : all_params['optim'],#sgd
              'use_scheduler' : False, #true decreaseing  
              'debug_mode'    : False 
    }
       
        print('Experiment with vars:  ',mod_opt_mlp)
        # print(mod_opt_xgb)
        options = {     'model_type'        : 'xgb', # xgb  mlp
                        'win_size'          : win_size,
                        'batch_size'        : 128,
                        'data'              : data,
                        'transform_targets' : True,
                        'split'             : 0.2,
                        'model_options' : mod_opt_mlp ,
                        'methods'       : None
        }
        options.update(mod_opt_mlp)

        exp.update_opt(options)
        res,loss = exp.train_and_test()
        print('accuracy: ',res,'loss: ',loss)
        return 1 - res




from skopt.plots import plot_convergence
dummy_results = skopt.dummy_minimize(search, SPACE,n_calls=20)
skopt.dump(dummy_results, 'results/dummy_mlp_hyper.pkl')
plt.figure(1)
plot_convergence(dummy_results)
plt.savefig("plots/dummy_mlp_hyper_accinv.png")

gp_results = skopt.gp_minimize(search, SPACE,n_calls=20)
skopt.dump(gp_results, 'results/gp_mlp_hyper.pkl')
plt.figure(2)
plot_convergence(gp_results)
plt.savefig("plots/gp_mlp_hyper_accinv.png")

gbrt_results = skopt.gbrt_minimize(search, SPACE, **HPO_PARAMS)
skopt.dump(gbrt_results, 'results/gbrt_mlp_hyper.pkl')
plt.figure(3)
plot_convergence(gbrt_results)
plt.savefig("plots/gbrt_mlp_hyper_accinv.png")
# print(results)


# plot_convergence(results)
# plt.savefig("plots/xgb_hyper_accinv.png")
results = [('random_results', dummy_results),
           ('gbrt_results', gbrt_results),
           ('gp_results', gp_results)]
plt.figure(4)
plot_convergence(*results)
plt.savefig("plots/all_mlp_hyper_accinv.png")
