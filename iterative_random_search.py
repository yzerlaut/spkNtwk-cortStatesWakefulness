import itertools
import numpy as np

from model import REC_POPS, AFF_POPS, Model

INITIAL_LIMS = [0,100]

GRID = np.zeros((len(REC_POPS)+len(AFF_POPS), len(REC_POPS),2))
INITIAL_GRID = 0*GRID
for i, j in itertools.product(range(len(REC_POPS)+len(AFF_POPS)), range(len(REC_POPS))):
    GRID[i,j,:] = np.array(INITIAL_LIMS)


def sort_random_configs(GRID, n=10000):

    CONFIGS = np.zeros((n,len(REC_POPS)+len(AFF_POPS), len(REC_POPS)))
    for i, j in itertools.product(range(len(REC_POPS)+len(AFF_POPS)), range(len(REC_POPS))):
        CONFIGS[:,i,j] = np.random.uniform(GRID[i,j,0], GRID[i,j,1], size=n)

    return CONFIGS

def run_sim(CONFIGS):

    RESIDUAL = np.ones(CONFIGS.shape[0])
    
    return RESIDUAL


def save_config(CONFIGS, RESIDUAL):

    np.save('data/Mscan/test.npy', np.array([CONFIGS, RESIDUAL]))
    
def launch_search():
    try:
        while True:
            print('!')
    except KeyboardInterrupt:
        print('Scan stopped')

if __name__=='__main__':
    # launch_search()

    CONFIGS = sort_random_configs(GRID, n=1000)
    RESIDUAL = run_sim(CONFIGS)
    save_config(CONFIGS, RESIDUAL)
    
