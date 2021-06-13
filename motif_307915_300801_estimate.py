# %% Import
import json 
import numpy as np
import argparse 
import pandas as pd

# %% Define parser
def ParseArguments():
    parser = argparse.ArgumentParser(description = "Motif generator")
    parser.add_argument('--input', default = "generated_data.json", required = False, help = 'Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default = "estimated_params.json", required = False, help = 'Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default = "no", required=False, help = 'Czy estymowac alpha czy nie?  (default: %(default)s)')
    args, _ = parser.parse_known_args()
    return args.input, args.output, args.estimate_alpha

# %% Define dtv
def dtv(th, thB, th0, thB0):
    w = th.shape[1]
    d = np.sum(np.abs(thB-thB0))
    for i in range(w):
        d += np.sum(np.abs(th[:,i]-th0[:,i]))
    return d/(w+1)

# %% Create initial distributions
def init_thetas(X, opt = 'random', **kwargs):
    '''Return Theta, ThetaB;
    X: data
    Theta: matrix 4 x w, ThetaB: vector of length w'''
    k,w = X.shape

    if opt == 'random':
        ThetaB = np.zeros(4)
        ThetaB[:(4-1)]=np.random.rand(4-1)/4
        ThetaB[4-1]=1-np.sum(ThetaB)
        
        # each column add up to 1
        Theta = np.zeros((4,w))
        Theta[0:3,:] = np.random.random((3,w))/w
        Theta[3,:] = 1 - np.sum(Theta, axis=0)

    elif opt == 'uniform':
        Theta = np.full((4,w), 1/4)
        ThetaB = np.array([1/4,1/4,1/4,1/4])

    elif opt == 'mean':
        Theta = np.full((4,w), 1/4)
        ThetaB = np.array([1/4,1/4,1/4,1/4])
        for j in range(w):
            counts = np.array([0,0,0,0])
            ls, cs = np.unique(X[:,j], return_counts = True)
            for i in range(len(ls)):
                counts[ls[i]-1] = cs[i]
            Theta[:,j] = counts / k
        ThetaB = np.mean(Theta, axis=1)

    else:
        raise ValueError(f'Invalid argument for function init_thetas: {opt}.\nPossible values: \'random\', \'uniform\', \'mean\'')
    
    if np.isnan(Theta).any() or np.isnan(ThetaB).any():
        raise ValueError
    
    return Theta, ThetaB

# %% Define EM
def EM(X, Theta, ThetaB, alpha, estimate_alpha):
    '''X: data; Theta, ThetaB, alpha: initial values (if estimate_alpha = True, alpha will be ignored)'''
    if np.isnan(Theta).any() or np.isnan(ThetaB).any():
        raise ValueError
    
    k,w = X.shape
    
    # Initial values
    if estimate_alpha: 
        alpha = 0.5 

    Theta_new = np.copy(Theta)
    ThetaB_new = np.copy(ThetaB)
    if np.isnan(Theta_new).any() or np.isnan(ThetaB_new).any():
        raise ValueError

    Q = np.full((2,k), 0, dtype = float)
    h = 0.00001 # convergence check treshold

    while True:
        if estimate_alpha: 
            alpha_new = lbd1 / np.sum(Q[0] + Q[1])
            # Prevent errors like true_division
            if alpha_new == 0: alpha_new = 0.00001
            elif alpha_new == 1: alpha_new = 0.99999
        
        Theta = np.copy(Theta_new)
        ThetaB = np.copy(ThetaB_new)
        if np.isnan(Theta).any() or np.isnan(ThetaB).any():
            raise ValueError

        # Expectation Step
        Q[0] = (1-alpha)*np.prod(ThetaB[X-1], axis=1)
        Q[1] = alpha*np.prod(Theta[X-1,np.arange(w)], axis=1)
        c = Q[0] + Q[1]
        Q = Q / c
        
        # Maximization step
        # Update alpha
        lbd1 = np.sum(Q[1])
        if estimate_alpha: 
            alpha_new = lbd1 / np.sum(Q[0] + Q[1])
            # Prevent errors like true_division
            if alpha_new == 0: alpha_new = 0.00001
            elif alpha_new == 1: alpha_new = 0.99999

        # Update thetas
        for l in [1,2,3,4]:
            Xl = X == l
            ThetaB_new[l-1] = np.sum(Q[0]*np.sum(Xl, axis=1))
            if np.isnan(ThetaB_new).any():
                raise ValueError
            for j in range(w):
                Theta_new[l-1,j] = np.sum(Q[1][Xl[:,j]])
                if np.isnan(Theta_new).any():
                    raise ValueError
        
        lbd0 = np.sum(ThetaB_new)
        try: 
            ThetaB_new = ThetaB_new / lbd0
        except Warning:
            raise
        try:
            Theta_new = Theta_new / lbd1   
        except Warning:
            raise 
        # Check if converges
        if dtv(Theta_new, ThetaB_new, Theta, ThetaB) < h: break

    Theta = Theta_new
    ThetaB = ThetaB_new
    if estimate_alpha: 
        alpha = alpha_new

    return Theta, ThetaB, alpha

# %% Define experiment
def run_experiment(X, init_opt, alpha, estimate_alpha, **kwargs):
    '''
    Run experiment and return estimated values and used parameters

    Arguments:
        X: data
        init_opt: random / uniform / mean (how to init values for EM?)
        estimate_alpha: if True, any passed alpha will be ignored and estimated
    
    Accepted kwargs:
        random_iter: number of iterations in random init method
    '''
    k,w = X.shape
    Theta, ThetaB = init_thetas(X, init_opt)
    est_Theta, est_ThetaB, est_alpha = EM(X, Theta, ThetaB, alpha, estimate_alpha)
    
    params = {
            'k': k,
            'w': w,
            'init': init_opt,
            'est_alpha': estimate_alpha
        }

    return est_alpha, est_Theta, est_ThetaB, params

# %% Program
if __name__ == '__main__':   
    input_file, output_file, estimate_alpha = ParseArguments()
    with open(input_file, 'r') as inputfile:
        data = json.load(inputfile)
    
    alpha = data['alpha']
    X = np.asarray(data['X'])

    # Run experiment
    est_a, est_Th, est_ThB, params = run_experiment(X, 'mean', alpha, estimate_alpha = True, random_iter = 1)

    # Extract results and save output
    estimated_params = {
        "alpha" : est_a,            
        "Theta" : est_Th.tolist(),   
        "ThetaB" : est_ThB.tolist()  
        }

    with open(output_file, 'w') as outfile:
        json.dump(estimated_params, outfile)
        print(f'Results saved in {output_file}.')