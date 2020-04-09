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

    # ## load file
    # data = ntwk.load_dict_from_hdf5('draft_data.h5')
    
    # # ## plot
    # fig, _ = ntwk.activity_plots(data,
    #                              smooth_population_activity=10)
    
    # ntwk.show()

elif sys.argv[-1]=='mf':

    mf = ntwk.FastMeanField(Model, REC_POPS, AFF_POPS)

    
    data = ntwk.load_dict_from_hdf5('draft_data.h5')
    tstop, dt = 1e-3*data['tstop'], 1e-2
    subsampling = int(dt/(1e-3*data['dt']))
    # subsampled t-axis
    t_sim = np.arange(int(data['tstop']/data['dt']))*data['dt']
    t = 1e-3*t_sim[::subsampling]

    DYN_SYSTEM, INPUTS = {}, {}
    for rec in REC_POPS:
        Model['COEFFS_%s' % rec] = np.load('data/COEFFS_pyrExc.npy')
        DYN_SYSTEM[rec] = {'aff_pops':['AffExc', 'NoiseExc'], 'x0':1e-2}
        INPUTS['AffExc_%s' % rec] = data['Rate_AffExc_%s' % rec][::subsampling]
        INPUTS['NoiseExc_%s' % rec] = data['Rate_NoiseExc_%s' % rec][::subsampling]
    
    CURRENT_INPUTS = {'oscillExc':Model['oscillExc_Ioscill_amp']*(1-np.cos(Model['oscillExc_Ioscill_freq']*2*np.pi*t))/2.}
    
    if not os.path.isfile('data/draft_mf_result.npz') or\
       input('Do you want to peform again the mean-field simulation ? [y/N]\n')=='y':
        print('performing calculus [...]')
        X = ntwk.mean_field.solve_mean_field_first_order(Model,
                                                         DYN_SYSTEM,
                                                         INPUTS=INPUTS,
                                                         CURRENT_INPUTS=CURRENT_INPUTS,
                                                         dt=dt, tstop=tstop)
        np.savez('data/draft_mf_result.npz', **X)
    else:
        X = load_dict('data/draft_mf_result.npz')
    
    ge = graph_env()
    fig, [ax0,ax1,ax2] = ge.figure(axes=(3,1), figsize=(3.,1.))

    ge.plot(t, Y=[INPUTS['AffExc_pyrExc'],
                  INPUTS['NoiseExc_pyrExc']],
            COLORS=['k', ge.brown],
            LABELS=['AffExc', 'NoiseExc'], ax=ax0,
            axes_args=dict(ylabel='rate (Hz)', xlim=[t[0], t[-1]]))
    ax0.legend(frameon=False, fontsize=ge.fontsize, loc='best')
    ge.annotate(ax0, 'network input', (0.,1.), bold=True)
    ge.plot(t,
            Y=[X['pyrExc']+1e-2, X['oscillExc'], X['recInh']+1e-2, X['dsInh']+1e-2],
            COLORS=[ge.g, ge.b, ge.r, ge.purple], ax=ax1,
            axes_args=dict(yticks=[0.1,1.,10], ylim=[.9e-2, 200], yscale='log', ylabel='rate (Hz)', xlim=[t[0], t[-1]]))
    ge.annotate(ax1, 'mean-field prediction', (0.,1.), bold=True)
    ge.plot(1e-3*t_sim,
            Y=[smooth(x, 100) for x in [data['POP_ACT_pyrExc']+1e-2, data['POP_ACT_oscillExc'], data['POP_ACT_recInh']+1e-2, data['POP_ACT_dsInh']+1e-2]], 
            COLORS=[ge.g, ge.b, ge.r, ge.purple], ax=ax2,
            axes_args=dict(yticks=[0.1,1.,10], ylim=[.9e-2, 200], yscale='log', ylabel='rate (Hz)', xlim=[t[0], t[-1]], xlabel='time (s)'))
    ge.annotate(ax2, 'numerical simulation', (0.,1.), bold=True)
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

