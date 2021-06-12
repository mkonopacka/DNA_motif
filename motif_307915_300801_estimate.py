# %% Import
import json 
import numpy as np
import argparse 
import pandas as pd

# %% Parse arguments
def ParseArguments():
    parser = argparse.ArgumentParser(description = "Motif generator")
    parser.add_argument('--input', default = "generated_data.json", required = False, help = 'Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default = "estimated_params.json", required = False, help = 'Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default = "no", required=False, help = 'Czy estymowac alpha czy nie?  (default: %(default)s)')
    args, _ = parser.parse_known_args()
    return args.input, args.output, args.estimate_alpha
    
input_file, output_file, estimate_alpha = ParseArguments()
with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)
 
alpha = data['alpha']
X = np.asarray(data['X'])
k,w = X.shape

# %% Define dtv
def dtv(th, thB, th0, thB0):
    d = np.sum(np.abs(thB-thB0))
    for i in range(w):
        print(f'{type(th0)} \n{th0}') # tu się coś psuje, th0 staje się listą
        d += np.sum(np.abs(th[:,i]-th0[:,i]))
    return d/(w+1)

# %% Create initial distributions
def init_thetas(w, opt = 'random', **kwargs):
    '''Return Theta, ThetaB; w(int): length of a sequence
    Theta: matrix 4 x w, ThetaB: vector of length w'''
    if opt == 'random':
        ThetaB = np.zeros(4)
        ThetaB[:(4-1)]=np.random.rand(4-1)/4
        ThetaB[4-1]=1-np.sum(ThetaB)
        Theta = np.zeros((4,w))
        Theta[:(w),:]=np.random.random((3,w))/w
        Theta[w,:]=1-np.sum(Theta, axis=0)

    elif opt == 'uniform':
        Theta = np.full((4,w), 1/4)
        ThetaB = np.array([1/4,1/4,1/4,1/4])

    elif opt == 'mean':
        Theta = np.full((4,w), 1/4)
        ThetaB = np.array([1/4,1/4,1/4,1/4])
        for j in range(w):
            _, counts = np.unique(X[:,j], return_counts = True)
            Theta[:,j] = counts / k
        ThetaB = np.mean(Theta, axis=1)

    else:
        raise ValueError(f'Invalid argument for function init_thetas: {opt}.\nPossible values: \'random\', \'uniform\', \'mean\'')
    
    return Theta, ThetaB

# %% Define EM
def EM(Theta, ThetaB, alpha, estimate_alpha = False):
    '''Theta, ThetaB, alpha: initial values (if estimate_alpha = True, alpha will be ignored)'''
    # Initial values
    if estimate_alpha: 
        alpha = 0.5 
        alpha_new = alpha

    Theta_new = np.copy(Theta)
    ThetaB_new = np.copy(ThetaB)

    Q = np.full((2,k), 0, dtype=float)
    h = 0.00001 # convergence check treshold

    while True:
        if estimate_alpha: alpha = alpha_new
        Theta = np.copy(Theta_new)
        ThetaB = np.copy(ThetaB_new)
        # Expectation Step
        Q[0] = (1-alpha)*np.prod(ThetaB[X-1], axis=1)
        Q[1] = alpha*np.prod(Theta[X-1,np.arange(w)], axis=1)
        c = Q[0] + Q[1]
        Q = Q / c
        
        # Maximization step
        # Update alpha
        lbd1 = np.sum(Q[1])
        if estimate_alpha: alpha_new = lbd1 / np.sum(Q[0] + Q[1])

        # Update thetas
        for l in [1,2,3,4]:
            Xl = X == l
            ThetaB_new[l-1] = np.sum(Q[0]*np.sum(Xl, axis=1))
            for j in range(w):
                Theta_new[l-1,j] = np.sum(Q[1][Xl[:,j]])
        
        lbd0 = np.sum(ThetaB_new)
        ThetaB_new = ThetaB_new / lbd0
        Theta_new = Theta_new / lbd1    
        # Check if converges
        print('INSIDE EM')
        if dtv(Theta_new, ThetaB_new, Theta, ThetaB) < h: break

    Theta = Theta_new
    ThetaB = ThetaB_new
    if estimate_alpha: alpha = alpha_new

    return Theta, ThetaB, alpha
# %% Define experiment
def run_experiment(w, init_opt, alpha, real_Theta, real_ThetaB, estimate_alpha = False, **kwargs):
    '''
    Run experiment and return summary dtf containing of:
        all passed arguments
        obtained dtv
        alpha_est - alpha_real difference

    Arguments:
        w: length of a sequence
        real_*: real values of parameters used to evaluate results
        init_opt: random / uniform / mean (how to init values for EM?)
        estimate_alpha: if True, any passed alpha will be ignored and estimated
    Accepted kwargs:
        random_iter: number of iterations in random init method
    '''
    if init_opt == 'random': 
        if not 'random_iter' in kwargs: raise ValueError('Unspecified number of iterations in random init method')
        iter = kwargs['random_iter']
        
        for i in range(iter):
            dtvs = []
            best_dtv = float('inf')
            Theta, ThetaB = init_thetas(w, init_opt)
            best_Theta, best_ThetaB, est_alpha = Theta, ThetaB, alpha
            Theta, ThetaB, alpha = EM(Theta, ThetaB, alpha, estimate_alpha)
            print('Inside init_opt == random!')
            new_dtv = dtv(Theta, ThetaB, real_Theta, real_ThetaB)
            dtvs.append(new_dtv)
            if new_dtv < best_dtv:
                best_dtv = new_dtv
                best_Theta = Theta
                best_ThetaB = ThetaB
                est_alpha = alpha
        
        obtained_dtv = best_dtv
        est_Theta = best_Theta
        est_ThetaB = best_ThetaB

        summary_method = {
            'random_dtv_min': min(dtvs),
            'random_dtv_avg': np.mean(dtvs),
            'random_no_iterations': iter,
            'best_random_Th0': best_Theta,
            'best_random_ThB0': best_ThetaB}

    summary = {
        'w': w,
        'init': init_opt,
        'real_alpha': alpha,
        'est_Theta': est_Theta,
        'est_ThetaB': est_ThetaB,
        'est_alpha': est_alpha,
        'dtv': obtained_dtv
    }

    return pd.DataFrame(data = {**summary, **summary_method})
            
# %% Read real_params for results evaluation
with open('params_set1.json', 'r') as real:
    real_data = json.load(real)
    real_Theta = real_data['Theta']
    real_ThetaB = real_data['ThetaB']
    real_alpha = real_data['alpha']

# %% Run experiment
results = run_experiment(w, 'random', alpha, real_Theta, real_ThetaB, random_iter = 1)

# %% Extract results and save output

estimated_params = {
    "alpha" : est_alpha,            # "przepisujemy" to alpha, one nie bylo estymowane 
    "Theta" : est_Theta.tolist(),   # westymowane
    "ThetaB" : est_ThetaB.tolist()  # westymowane
    }

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
    print(f'Results saved in {output_file}.')
    

# TODO ZAUTOMATYZOWAĆ TESTY DLA: 
# podaję parametry (wygenerować je)
# dla danego zestawu parametrów i sposobu losowania 
# przeprowadza test kilka razy i wyciąga średnią z dtv
