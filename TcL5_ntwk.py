"""

"""
import sys
sys.path.append('./neural_network_dynamics')

import numpy as np
import matplotlib.pylab as plt
import ntwk

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

Model = {
    ## ---------------------------------------------------------------------------------
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_TcExc':200, 
    'N_ReInh':200, 
    'N_L5Exc':1000, 
    'N_BgExc':100,
    'N_ModExc':100,
    # synaptic weights
    'Q_TcExc_L5Exc':8., 
    'Q_TcExc_ReInh':5., 
    'Q_ReInh_TcExc':15., 
    'Q_ReInh_ReInh':15., 
    'Q_L5Exc_TcExc':5., 
    'Q_BgExc_TcExc':10.,
    'Q_ModExc_L5Inh':5.,
    'Q_ModExc_ReInh':5.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # connectivity parameters
    'p_TcExc_L5Exc':0.05, 
    'p_TcExc_ReInh':0.05, 
    'p_L5Exc_TcExc':0.015, 
    'p_ReInh_ReInh':0.08, 
    'p_ReInh_TcExc':0.3, 
    'p_BgExc_TcExc':0.1, 
    'p_ModExc_ReInh':0.2, 
    # simulation parameters
    'dt':0.1, 'tstop': 4000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-70., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 70, 'TcExc_tauw':200.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':3.,
    'ReInh_El':-70., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':50., 'ReInh_b': 30, 'ReInh_tauw':200.,
    # --> Layer 4 cortical population (Exc)
    'L5Exc_Gl':10., 'L5Exc_Cm':200.,'L5Exc_Trefrac':3.,
    'L5Exc_El':-70., 'L5Exc_Vthre':-50., 'L5Exc_Vreset':-60., 'L5Exc_deltaV':2.5,
    'L5Exc_a':0., 'L5Exc_b': 100, 'L5Exc_tauw':200.,
    #
    'REC_POPS':['TcExc', 'ReInh', 'L5Exc'],
    'AFF_POPS':['BgExc', 'ModExc'],
}

t = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_BgExc'] = ntwk.stim.waveform_library.varying_levels_function(\
                                                        t, [0, 20], [0, 300], 200)
Model['Farray_ModExc'] = ntwk.stim.waveform_library.varying_levels_function(\
                                                        t, [0, 10], [0, 2e3], 200)


if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        ntwk.quick_run.simulation(Model, with_Vm=3, verbose=False,
                                  filename='data/thal-oscill.ntwk.h5')

    data = ntwk.recording.load_dict_from_hdf5('data/thal-oscill.ntwk.h5')

    fig, _ = ntwk.plots.activity_plots(data, 
                                       pop_act_args=dict(smoothing=10, 
                                                         subsampling=2, 
                                                         log_scale=False),
                                       fig_args=dict(figsize=(3,0.7), dpi=75))

    # fig, ax = ntwk.plots.pt.figure(figsize=(4,12), top=0, left=0, bottom=0)
    # ntwk.plots.few_Vm_plot(data, lw=0.5, vpeak=10, shift=100,
                           # spike_style='-', ax=ax,
                           # COLORS=[ntwk.plots.pt.tab10(i) for i in range(4)],
                           # NVMS=[[0] for i in Model['REC_POPS']])

    plt.show()
