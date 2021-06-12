# %% Import libraries
import json 
import numpy as np
import argparse 

# %% Read parameters
def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    args, _ = parser.parse_known_args()
    return args.params, args.output

def generate_data(param_file, output_file):
    with open(param_file, 'r') as inputfile:
        params = json.load(inputfile)
    
    w = params['w']
    k = params['k']
    alpha = params['alpha']
    Theta = np.asarray(params['Theta'])
    ThetaB = np.asarray(params['ThetaB'])

    X = np.full((k,w), 0)

    for i in range(k):
        if np.random.random() < alpha:
            for j in range(w):
                probs = Theta[:,j]
                X[i,j] = np.random.choice(4, 1, p = probs) + 1
        else:
            X[i,:] = np.random.choice(4, w, p = ThetaB) + 1

    gen_data = {    
        "alpha" : alpha,
        "X" : X.tolist()
        }

    with open(output_file, 'w') as outfile:
        json.dump(gen_data, outfile)
    
    print(f'New data generated and stored in {output_file}.')
 
if __name__ == '__main__':  
    param_file, output_file = ParseArguments()
    generate_data(param_file, output_file)
