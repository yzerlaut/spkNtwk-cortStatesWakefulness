import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import neural_network_dynamics.main as ntwk

from analyz.processing.signanalysis import gaussian_smoothing as smooth
from analyz.IO.npz import load_dict
from datavyz.main import graph_env

from model import Model, REC_POPS, AFF_POPS

if sys.argv[-1]=='plot':


    
    ######################
    ## ----- Plot ----- ##
    ######################

    ## load file
    data = ntwk.load_dict_from_hdf5('draft_data.h5')
    
    # ## plot
    fig, _ = ntwk.activity_plots(data,
                                 smooth_population_activity=10)
    
    ntwk.show()

elif sys.argv[-1]=='mf':

    ge = graph_env()
    
    mf = ntwk.FastMeanField(Model, REC_POPS, AFF_POPS)
    mf.build_TF_func(100)
    X = mf.run_single_connectivity_sim(mf.ecMatrix)
    
    COLORS=[ge.g, ge.b, ge.r, ge.purple]
    
    data = ntwk.load_dict_from_hdf5('draft_data.h5')
    fig, AX = ntwk.activity_plots(data,
                                  COLORS=COLORS,
                                  smooth_population_activity=10)

    for i, label in enumerate(REC_POPS):
        AX[-1].plot(1e3*mf.t, X[:,i], lw=4, color=COLORS[i], alpha=.5)
    
    ge.show()

else:


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
    ntwk.write_as_hdf5(NTWK, filename='draft_data.h5')
    print('Results of the simulation are stored as:', 'draft_data.h5')
    print('--> Run \"python draft.py plot\" to plot the results')

