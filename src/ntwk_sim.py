import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import neural_network_dynamics.main as ntwk

from analyz.processing.signanalysis import gaussian_smoothing as smooth
from analyz.IO.npz import load_dict

# from Umodel import Umodel

from datavyz import ge
COLORS=[ge.green, ge.blue, ge.red, ge.purple, 'k', 'dimgrey']


#######################################
#####  RUNNING SIMULATIONS ############
#######################################

def run_sim(Model,
            filename='draft_data.h5', verbose=True):

    ######################
    ## ----- Run  ----- ##
    ######################

    REC_POPS, AFF_POPS = list(Model['REC_POPS']), list(Model['AFF_POPS'])
    # Model['tstop'], Model['dt'] = 6000, 0.1

    print('--initializing simulation for %s [...]' % filename)

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    faff =  ntwk.stim_waveforms.IncreasingSteps(t_array, 'AffExc', Model, translate_to_SI=False)
    fnoise = Model['F_NoiseExc']+0*t_array

    #######################################
    ########### BUILD POPS ################
    #######################################

    NTWK = ntwk.build_populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  with_pop_act=True,
                                  with_raster=True,
                                  with_Vm=4,
                                  verbose=verbose)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=verbose)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    #  time-dep afferent excitation
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t_array, faff,
                                         verbose=verbose,
                                         SEED=4)
    # # noise excitation
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'NoiseExc',
                                         t_array, fnoise,
                                         verbose=verbose,
                                         SEED=5)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    print('----   running simulation for %s [...]' % filename)
    network_sim = ntwk.collect_and_run(NTWK, verbose=verbose)

    #####################
    ## ----- Save ----- ##
    #####################
    ntwk.write_as_hdf5(NTWK, filename=filename)
    print('[ok] Results of the simulation are stored as:', filename)


#############################################
#####  RUNNING (SLOW) MEAN_FIELD ############
#############################################

def run_slow_mf(filename):
    """
    always respective to a spiking network num. sim
    """

    print('--   running (slow) mean-field for %s [...]' % filename)
    data = ntwk.load_dict_from_hdf5(filename)
    Model = data
        
    tstop, dt = 1e-3*data['tstop'], 1e-2
    subsampling = int(dt/(1e-3*data['dt']))
    # subsampled t-axis
    t_sim = np.arange(int(data['tstop']/data['dt']))*data['dt']
    t = 1e-3*t_sim[::subsampling]

    DYN_SYSTEM, INPUTS = {}, {}
    for rec in list(data['REC_POPS']):
        Model['COEFFS_%s' % rec] = np.load('data/COEFFS_pyrExc.npy')
        DYN_SYSTEM[rec] = {'aff_pops':list(data['AFF_POPS']), 'x0':1e-2}
        INPUTS['AffExc_%s' % rec] = data['Rate_AffExc_%s' % rec][::subsampling]
        INPUTS['NoiseExc_%s' % rec] = data['Rate_NoiseExc_%s' % rec][::subsampling]
    
    CURRENT_INPUTS = {'oscillExc':Model['oscillExc_Ioscill_amp']*(1-np.cos(Model['oscillExc_Ioscill_freq']*2*np.pi*t))/2.}
    
    X = ntwk.mean_field.solve_mean_field_first_order(Model,
                                                         DYN_SYSTEM,
                                                         INPUTS=INPUTS,
                                                         CURRENT_INPUTS=CURRENT_INPUTS,
                                                         dt=dt, tstop=tstop)

    np.savez(filename.replace('.ntwk.h5', '.mf.npz'), **X)

    
#######################################
##### PLOTTING SIMULATIONS ############
#######################################
    
def plot_sim(filename, ge,
             omf_data=None,
             Umodel_data=None,
             mf_data=None):
    
    ## load file
    data = ntwk.load_dict_from_hdf5(filename)
    
    # ## plot
    fig, AX = ntwk.activity_plots(data,
                                  smooth_population_activity=10,
                                  COLORS=COLORS,
                                  pop_act_log_scale=True)

    if Umodel_data is not None:
        AX[2].plot(Umodel_data['t'],
                   Umodel_data['desired_Vm'], '-', lw=2, color='dimgrey', label='mean-field')

    if mf_data is None:

        mf = ntwk.FastMeanField(data, tstop=6., dt=2.5e-3, tau=20e-3)
        mf.build_TF_func(tf_sim_file='neural_network_dynamics/theory/tf_sim_points.npz')
        X, mVm, sVm = mf.run_single_connectivity_sim(mf.ecMatrix, verbose=True)
        t = np.linspace(0, data['tstop'], X.shape[1])
        for i, label in enumerate(data['REC_POPS']):
            AX[-1].plot(t, 1e-2+X[i,:], '--', lw=1, color=COLORS[i])
            ge.plot(t, 1e3*mVm[i,:], sy=1e3*sVm[i,:], color='k', lw=1, label='mean-field', ax=AX[2+i])
            AX[2+i].plot(t, 1e3*(mVm[i,:]-sVm[i,:]), 'k-', lw=0.5)
            AX[2+i].plot(t, 1e3*(mVm[i,:]+sVm[i,:]), 'k-', lw=0.5)
        # AX[2].plot(mf_data['t'], mf_data['desired_Vm'], 'k--', lw=2, label='U-model')
        # AX[2].legend(frameon=False, loc='best')

        
    if omf_data is None:
        try:
            omf_data = load_dict(filename.replace('ntwk.h5', 'mf.npz'))
        except FileNotFoundError:
            pass
    if omf_data is not None:
        t = np.linspace(0, data['tstop'], len(omf_data['pyrExc']))
        for i, label in enumerate(data['REC_POPS']):
            AX[-1].plot(t, 1e-2+omf_data[label], '-', lw=4, color=COLORS[i], alpha=.5)
            # AX[-1].plot(t, omf_data[label], 'k--')
        
    figM, _, _ = plot_matrix(data)
    
    ntwk.show()
    # ge.show()
    

def plot_matrix(Model, ge=None):

    REC_POPS, AFF_POPS = list(Model['REC_POPS']), list(Model['AFF_POPS'])

    pconnMatrix = np.zeros((len(REC_POPS)+len(AFF_POPS), len(REC_POPS)))
    
    for i, source_pop in enumerate(REC_POPS+AFF_POPS):
        for j, target_pop in enumerate(REC_POPS):
            pconnMatrix[i,j] = Model['p_%s_%s' % (source_pop, target_pop)]

    if ge is None:
        from datavyz import ge
        
    fig, ax = ge.figure(right=5.)
    
    Lims = [np.round(100*pconnMatrix.min(),1)-.1,np.round(100*pconnMatrix.max(),1)+.1]
    
    ge.matrix(100*pconnMatrix.T, origin='upper',
              ax=ax, vmin=Lims[0], vmax=Lims[1])


    n, m = len(REC_POPS)+len(AFF_POPS), len(REC_POPS)
    for i, label in enumerate(REC_POPS+AFF_POPS):
        ge.annotate(ax, label, (-0.2, .9-i/n),\
            ha='right', va='center', color=COLORS[i])
    for i, label in enumerate(REC_POPS):
        ge.annotate(ax, label, (i/m+.25, -0.1),\
                    ha='right', va='top', color=COLORS[i], rotation=65)
    
    acb = ge.bar_legend(fig,
                        inset=dict(rect=[.72,.3,.03,.5], facecolor=None),
                        colormap=ge.viridis,
                        bounds=[Lims[0], Lims[1]],
                        label='$p_{conn}$ (%)')
    ge.set_plot(ax,
                ['left', 'bottom'],
                tck_outward=0,
                xticks=.75*np.arange(0,4)+.75/2.,
                xticks_labels=[],
                xlim_enhancment=0, ylim_enhancment=0,
                yticks=.83*np.arange(0,6)+.85/2.,
                yticks_labels=[])

    
    return fig, ax, acb
    
    
if __name__=='__main__':

    if sys.argv[-1]=='mf':
        run_slow_mf(sys.argv[-2])
    else:
        plot_sim(sys.argv[-1], ge)
    
