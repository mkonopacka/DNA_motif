#%%
import json
import numpy as np
from numpy.lib.npyio import save
from motif_307915_300801_generate import generate_data
from motif_307915_300801_estimate import *
from json_save_params import *
import pandas as pd
from collections import OrderedDict

def run_test(init_opt = 'mean', est_alpha = True, param_file = 'params_set.json'):
    print('New test started -------------------')
    # 1. Read real data
    print('Reading params from file ...')
    with open(param_file, 'r') as real:
        real_data = json.load(real)
        real_Theta = np.array(real_data['Theta'])
        real_ThetaB = np.array(real_data['ThetaB'])
        real_alpha = np.array(real_data['alpha'])
    
    # 2. Generate data from real params and save it in generated_data.json
    generate_data(param_file, 'generated_data.json')

    # 3. Run experiment on generated data
    with open('generated_data.json', 'r') as inputfile:
        data = json.load(inputfile)

    X = np.asarray(data['X'])

    est_a, est_Th, est_ThB, params = run_experiment(X, init_opt, real_alpha, estimate_alpha = est_alpha)
    result = dtv(est_Th, est_ThB, real_Theta, real_ThetaB)

    results_summary = {
        'dtv': result,
        'alpha_dif': est_a - real_alpha
    }

    return {**params,**results_summary}

if __name__ == '__main__':
    # Test data: create some Theta (4 x w), ThetaB pairs
    # len(ThetaB) == Theta.shape[1]
    # Each Theta_pair = (Theta, ThetaB, Id) 
    Thetas0 = (
        np.array([[3/8,1/8,2/8,2/8],[1/10,2/10,3/10,4/10],[1/7,2/7,1/7,3/7]]).T, 
        np.array([1/4,1/4,1/4,1/4]), 
        0
    )
    Theta_pairs = [Thetas0]
    
    # Add csv results file header
    with open('test_results.csv', 'a') as f:
        f.write('k,w,init,est_alpha,dtv,alpha_dif,Thetas_ID\n')
    
    for pair in Theta_pairs:
        # different k
        for k in [3,10,100,200,1000,2000]:
            # different alphas
            for alpha in [0.1,0.3,0.5,0.7]:
                # First create new params set
                save_params_to_json(outfile = 'params_set.json', Theta = pair[0], ThetaB = pair[1], alpha = alpha, k = k)
                # Then run test with generated params file
                for method in ['random', 'uniform', 'mean']:
                    for est_alpha in [True, False]:
                        result = OrderedDict(**run_test(method, est_alpha, param_file = 'params_set.json'), **{'Thetas_ID': pair[2]})
                        # Convert results to csv format line
                        line = ''
                        for key in result:
                            line += f'{result[key]},'
                        # Append result to test_results file
                        with open('test_results.csv', 'a') as f:
                            f.write(line[:-1]+'\n')

    # Convert to dataframe
    dtf = pd.read_csv('test_results.csv')

# %% Analyse results
dtf.head()