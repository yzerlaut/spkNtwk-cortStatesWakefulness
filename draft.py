import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import neural_network_dynamics.main as ntwk
import numpy as np

from model import Model, REC_POPS, AFF_POPS


# -------------------------------
# --- connectivity parameters ---
# -------------------------------
# ==> Exc
Model['p_Exc_Exc'] = 0.02
Model['p_Exc_PvInh'] = 0.05
Model['p_Exc_SstInh'] = 0.02
# ==> oscillExc
Model['p_oscillExc_oscillExc'] = 0.02
Model['p_oscillExc_Exc'] = 0.05
Model['p_oscillExc_PvInh'] = 0.05
# ==> PvInh
Model['p_PvInh_PvInh'] = 0.1
Model['p_PvInh_SstInh'] = 0.05
Model['p_PvInh_VipInh'] = 0.1
Model['p_PvInh_Exc'] = 0.05
Model['p_PvInh_oscillExc'] = 0.05
# ==> SstInh
# Model['p_SstInh_Exc'] = 0.05
# Model['p_SstInh_PvInh'] = 0.05
Model['p_SstInh_oscillExc'] = 0.05
# ==> VipInh
Model['p_VipInh_SstInh'] = 0.05
Model['p_VipInh_PvInh'] = 0.05
Model['p_VipInh_oscillExc'] = 0.05
# ==> AffExc
Model['p_AffExc_VipInh'] = 0.2
Model['p_AffExc_PvInh'] = 0.2
Model['p_AffExc_Exc'] = 0.05
Model['p_AffExc_oscillExc'] = 0.05
# ==> NoiseExc
Model['p_NoiseExc_SstInh'] = 0.2
Model['p_NoiseExc_PvInh'] = 0.02
Model['p_NoiseExc_VipInh'] = 0.02
Model['p_NoiseExc_oscillExc'] = 0.05
Model['p_NoiseExc_Exc'] = 0.01


if sys.argv[-1]=='plot':

    ######################
    ## ----- Plot ----- ##
    ######################

    ## load file
    data = ntwk.load_dict_from_hdf5('draft_data.h5')
    print(data['p_Exc_Exc'])
    
    # ## plot
    fig, _ = ntwk.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    ntwk.show()

else:

    Model['tstop'], Model['dt'] = 5000, 0.1
    t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    faff = np.array([3*int(tt/1000) for tt in t])

    #######################################
    ########### BUILD POPS ################
    #######################################
    
    NTWK = ntwk.build_populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  with_raster=True,
                                  with_Vm=4,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    #  time-dep afferent excitation
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t, faff,
                                         verbose=True,
                                         SEED=4)

    fnoise = 2.
    # noise excitation
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'NoiseExc',
                                         t, fnoise+0.*t,
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

