import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
# sys.path.append(os.path.join(str(pathlib.Path(__file__).resolve().parent), 'neural_network_dynamics'))
from neural_network_dynamics import ntwk

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

Model = {
    ## ---------------------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_RecExc':4000, 'N_RecInh':1000, 'N_AffExcBG':100, 'N_AffExcTV':100, 'N_DsInh':500,
    # synaptic weights (nS)
    'Q_RecExc_RecExc':2., 'Q_RecExc_RecInh':2., 
    'Q_RecInh_RecExc':10., 'Q_RecInh_RecInh':10., 
    'Q_AffExcBG_RecExc':4., 'Q_AffExcBG_RecInh':4., 'Q_AffExcBG_DsInh':4.,
    'Q_AffExcTV_RecExc':4., 'Q_AffExcTV_RecInh':4., 'Q_AffExcTV_DsInh':4.,
    'Q_DsInh_RecInh':10., 
    # synaptic time constants (ms)
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials (mV)
    'Ee':0., 'Ei': -80.,
    # connectivity parameters (proba.)
    'p_RecExc_RecExc':0.05, 'p_RecExc_RecInh':0.05, 
    'p_RecInh_RecExc':0.05, 'p_RecInh_RecInh':0.05, 
    'p_DsInh_RecInh':0.05, 
    'p_AffExcBG_RecExc':0.1, 'p_AffExcBG_RecInh':0.1, 'p_AffExcBG_DsInh':0.075,
    'p_AffExcTV_RecExc':0.1, 'p_AffExcTV_RecInh':0.1, 'p_AffExcTV_DsInh':0.075,
    # complex pattern stimulation
    'STIM-delay':400,
    'STIM-synchrony':0.2,
    'STIM-frequency':3, # Hz
    'STIM-width':10, # ms
    'STIM-SEED':3, # stimulus seed
    # backrgound afferent stimulation (Hz)
    'F_AffExcBG':4.,
    'BG-SEED':3, # background activity seed
    # simulation parameters (ms)
    'dt':0.1, 'tstop':4000,
    'SEED':3, # connectivity seed
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (RecExc, recurrent excitation)
    'RecExc_Gl':10., 'RecExc_Cm':200.,'RecExc_Trefrac':5.,
    'RecExc_El':-70., 'RecExc_Vthre':-50., 'RecExc_Vreset':-70., 'RecExc_delta_v':0.,
    'RecExc_a':0., 'RecExc_b': 0., 'RecExc_tauw':1e9, 'RecExc_deltaV':0,
    # --> Inhibitory population (RecInh, recurrent inhibition)
    'RecInh_Gl':10., 'RecInh_Cm':200.,'RecInh_Trefrac':5.,
    'RecInh_El':-70., 'RecInh_Vthre':-53., 'RecInh_Vreset':-70., 'RecInh_delta_v':0.,
    'RecInh_a':0., 'RecInh_b': 0., 'RecInh_tauw':1e9, 'RecInh_deltaV':0,
    # --> Disinhibitory population (DsInh, disinhibition)
    'DsInh_Gl':10., 'DsInh_Cm':200.,'DsInh_Trefrac':5.,
    'DsInh_El':-70., 'DsInh_Vthre':-50., 'DsInh_Vreset':-70., 'DsInh_delta_v':0.,
    'DsInh_a':0., 'DsInh_b': 0., 'DsInh_tauw':1e9, 'DsInh_deltaV':0,
    ## ---------------------------------------------------------------------------------
}

def run_single_sim(Model, verbose=False):

    StimPattern = {'indices':[], 'times':[]}
    Nrecruited = int(Model['STIM-synchrony']*Model['N_AffExcTV'])

    np.random.seed(Model['STIM-SEED'])

    events = Model['STIM-delay']+np.cumsum(np.random.exponential(1000/Model['STIM-frequency'], 
                        int(10*Model['STIM-frequency']*Model['tstop']/1000)))
    for event in events[events<Model['tstop']]:
        StimPattern['times'] += list(event+np.random.randn(Nrecruited)*Model['STIM-width'])
        StimPattern['indices'] += list(np.random.choice(range(Model['N_AffExcTV']), Nrecruited))

    REC_POPS = ['RecExc', 'RecInh', 'DsInh']
    AFF_POPS = ['AffExcBG', 'AffExcTV']

    NTWK = ntwk.build.populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  with_raster=True, 
                                  with_Vm=1,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=verbose)

    ntwk.build.recurrent_connections(NTWK, SEED=5, verbose=verbose)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']

    # background activity
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExcBG',
                                              t_array, Model['F_AffExcBG']+0.*t_array,
                                              verbose=verbose,
                                              SEED=int(Model['BG-SEED']))

    # build connectivity matrices for the stimulus
    ntwk.build.fixed_afference(NTWK, ['AffExcTV'], REC_POPS)

    # stimulus activity
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExcTV',
                                              t_array, 0.*t_array, # no background aff
                                              additional_spikes_in_terms_of_pre_pop=StimPattern,
                                              verbose=verbose,
                                              SEED=int(Model['STIM-SEED'])+1)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.build.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=verbose)

    ######################
    ## ----- Write ---- ##
    ######################

    NTWK['AffExcTV_indices'] = StimPattern['indices']
    NTWK['AffExcTV_times'] = StimPattern['times']

    return NTWK

if sys.argv[-1]=='plot':

    ######################
    ## ----- Plot ----- ##
    ######################
    
    ## load file
    data = ntwk.recording.load_dict_from_hdf5('data/complex_pattern.h5')

    # plot input patterm
    fig, ax = plt.subplots(1)
    ax.set_title('stim. pattern')
    ax.plot(data['AffExcTV_times'], data['AffExcTV_indices'], 'ko', ms=1)
    ax.set_xlim([0, data['tstop']])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('nrn ID')

    # ## plot
    fig, _ = ntwk.plots.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    plt.show()

elif sys.argv[-1]=='scan':

    Model['data_folder'] = './data/'
    Model['zip_filename'] = 'data/regime-seed-scan.zip'

    ntwk.scan.run(Model,
                  ['F_AffExcBG', 'BG-SEED'],
                  [np.array([4,20]),
                   np.arange(10)],
                  run_single_sim,
                  fix_missing_only=True,
                  parallelize=True)


else:

    #####################################
    ## ----- Example Simulation ------ ##
    #####################################

    NTWK = run_single_sim(Model)
    ntwk.recording.write_as_hdf5(NTWK, 
                                 ARRAY_KEYS=['AffExcTV_indices', 'AffExcTV_times'],
                                 filename='data/complex_pattern.h5')

    print('Results of the simulation are stored as:', 'data/complex_pattern.h5')
    print('--> Run \"python pattern_input_data.py plot\" to plot the results')
    
