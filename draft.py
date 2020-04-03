import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import neural_network_dynamics.main as ntwk
import numpy as np
from analyz.processing.signanalysis import smooth
from model import Model, REC_POPS, AFF_POPS


if sys.argv[-1]=='plot':

    ######################
    ## ----- Plot ----- ##
    ######################

    ## load file
    data = ntwk.load_dict_from_hdf5('draft_data.h5')
    
    # ## plot
    fig, _ = ntwk.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    ntwk.show()

else:

    ######################
    ## ----- Run  ----- ##
    ######################
    
    Model['tstop'], Model['dt'] = 6000, 0.1
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    faff = smooth(np.array([4*int(tt/1000) for tt in t_array]), int(200/0.1))
    
    fnoise = 3.

    #######################################
    ########### BUILD POPS ################
    #######################################
    
    NTWK = ntwk.build_populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
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
                                         t_array, fnoise+0.*t_array,
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
    
    ntwk.write_as_hdf5(NTWK, filename='draft_data.h5')
    print('Results of the simulation are stored as:', 'draft_data.h5')
    print('--> Run \"python draft.py plot\" to plot the results')

