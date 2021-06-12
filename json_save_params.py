import json 
import numpy as np
 
def save_params_to_json(outfile, Theta, ThetaB, alpha = 0.5, k = 1000):
    
    params = {
        "w" : Theta.shape[1], # matrix 4 x w
        "alpha" : alpha,
        "k" : k, # desired number of observations
        "Theta" : Theta.tolist(),
        "ThetaB" : ThetaB.tolist()
        }

    with open(outfile, 'w') as f:
        json.dump(params, f)
    
    print(f'Params saved in json format in file f{outfile}')

if __name__ == '__main__':
    # Example params
    tmp = np.array(
        [[3/8,1/8,2/8,2/8],
        [1/10,2/10,3/10,4/10],
        [1/7,2/7,1/7,3/7]])

    Theta = tmp.T
    ThetaB = np.array([1/4,1/4,1/4,1/4])
    save_params_to_json('params_set1.json', Theta, ThetaB)