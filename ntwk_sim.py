import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import neural_network_dynamics.main as ntwk

from analyz.processing.signanalysis import gaussian_smoothing as smooth
from analyz.IO.npz import load_dict
from datavyz.main import graph_env
ge = graph_env()
COLORS=[ge.g, ge.b, ge.r, ge.purple]

from model import Model, REC_POPS, AFF_POPS
from Umodel import Umodel


def run_sim(Model, filename='draft_data.h5'):

    ######################
    ## ----- Run  ----- ##
    ######################
    
    Model['tstop'], Model['dt'] = 6000, 0.1

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    faff =  ntwk.stim_waveforms.IncreasingSteps(t_array, 'AffExc', Model, translate_to_SI=False)
    fnoise = 3.+0*t_array

    #######################################
    ########### BUILD POPS ################
    #######################################
    print(Model)
    NTWK = ntwk.build_populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  with_pop_act=True,
                                  with_raster=True,
                                  with_Vm=4,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    #  time-dep afferent excitation
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t_array, faff,
                                         verbose=True,
                                         SEED=4)
    # # noise excitation
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'NoiseExc',
                                         t_array, fnoise,
                                         verbose=True,
                                         SEED=5)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)

    #####################
    ## ----- Save ----- ##
    #####################
    ntwk.write_as_hdf5(NTWK, filename=filename)
    print('Results of the simulation are stored as:', filename)
    print('--> Run \"python draft.py plot\" to plot the results')

    
def plot_sim(filename):
    ######################
    ## ----- Plot ----- ##
    ######################

    ## load file
    data = ntwk.load_dict_from_hdf5(filename)
    
    # ## plot
    fig, _ = ntwk.activity_plots(data,
                                 smooth_population_activity=10)
    
    ntwk.show()

    # um = Umodel()
    # AX[2].plot(1e3*mf.t, um.predict_Vm(mf.t, mf.FAFF[0,:])+Model['pyrExc_El'], 'k--')
    ge.show()
    um = Umodel()
    ge.plot(um.predict_Vm(1e-3*t_sim, data['Rate_AffExc_pyrExc']), fig_args={'figsize':(3,1)})
            
    ge.show()
    
    
    for i, label in enumerate(REC_POPS):
        AX[-1].plot(1e3*mf.t, 1e-2+X[i,:], lw=4, color=COLORS[i], alpha=.5)
        Vm = mf.convert_to_mean_Vm_trace(X, label, verbose=True)
        AX[i+2].plot(1e3*mf.t, 1e3*Vm, 'k-')
        AX[i+2].set_ylim([-72,-45])
        
if __name__=='__main__':

    try:
        plot_sim(sys.argv[-1])
    except BaseException:
        pass
    
