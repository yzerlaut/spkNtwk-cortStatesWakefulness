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


if sys.argv[-1]=='plot':

    ######################
    ## ----- Plot ----- ##
    ######################

    ## load file
    print('plotting "data/draft_data.h5" [...]')
    data = ntwk.load_dict_from_hdf5('data/draft_data.h5')
    
    # ## plot
    fig, _ = ntwk.activity_plots(data,
                                 smooth_population_activity=10)
    
    ntwk.show()

elif sys.argv[-1]=='mf':
    ################################
    ## -- Mean-Field (fast) ----- ##
    ################################

    mf = ntwk.FastMeanField(Model, REC_POPS, AFF_POPS, tstop=6.)

    mf.build_TF_func(100, with_Vm_functions=True, sampling='log')
    X, Vm = mf.run_single_connectivity_sim(mf.ecMatrix, verbose=True)

    
    data = ntwk.load_dict_from_hdf5('data/draft_data.h5')
    fig, AX = ntwk.activity_plots(data,
                                  COLORS=COLORS,
                                  smooth_population_activity=10,
                                  pop_act_log_scale=True)

    um = Umodel()
    AX[2].plot(1e3*mf.t, um.predict_Vm(mf.t, mf.FAFF[0,:])+Model['pyrExc_El'], 'k--')
    
    for i, label in enumerate(REC_POPS):
        AX[-1].plot(1e3*mf.t, 1e-2+X[i,:], lw=4, color=COLORS[i], alpha=.5)
        AX[i+2].plot(1e3*mf.t, 1e3*Vm[i,:], 'k-')
        AX[i+2].set_ylim([-72,-45])
        
    
    ge.show()

elif sys.argv[-1]=='old-mf':
    #########################
    ## -- Mean-Field ----- ##
    #########################

    data = ntwk.load_dict_from_hdf5('data/draft_data.h5')
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
    
    fig, AX = ntwk.activity_plots(data,
                                  COLORS=COLORS,
                                  smooth_population_activity=10,
                                  pop_act_log_scale=True)

    for i, pop in enumerate(REC_POPS):
        AX[-1].plot(1e3*t, 1e-2+X[pop], color=COLORS[i], lw=3, alpha=0.6)


    um = Umodel()
    ge.plot(um.predict_Vm(1e-3*t_sim, data['Rate_AffExc_pyrExc']), fig_args={'figsize':(3,1)})
            
    ge.show()
    

        
    ge.show()
    
else:
    from model import Model, REC_POPS, AFF_POPS
    from ntwk_sim import run_sim
    
    run_sim(Model, REC_POPS, AFF_POPS, filename='data/draft_data.h5')


