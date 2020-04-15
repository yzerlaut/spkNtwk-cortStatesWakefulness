import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import neural_network_dynamics.main as ntwk

from analyz.processing.signanalysis import gaussian_smoothing as smooth
from analyz.IO.npz import load_dict

# from Umodel import Umodel

from datavyz import ge
COLORS=[ge.g, ge.b, ge.r, ge.purple, 'k', 'dimgrey']


def run_sim(Model, REC_POPS, AFF_POPS,
            filename='draft_data.h5', verbose=True):

    ######################
    ## ----- Run  ----- ##
    ######################
    
    # Model['tstop'], Model['dt'] = 6000, 0.1

    print('initializing simulation for %s [...]' % filename)

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
    print('running simulation for %s [...]' % filename)
    network_sim = ntwk.collect_and_run(NTWK, verbose=verbose)

    #####################
    ## ----- Save ----- ##
    #####################
    ntwk.write_as_hdf5(NTWK, filename=filename)
    print('[ok] Results of the simulation are stored as:', filename)

    
def plot_sim(filename, ge):
    ######################
    ## ----- Plot ----- ##
    ######################
    from model import REC_POPS, AFF_POPS
    ## load file
    data = ntwk.load_dict_from_hdf5(filename)
    # for key in Model:
    #     print(key, '-->', data[key])
    # ## plot
    fig, AX = ntwk.activity_plots(data,
                                  smooth_population_activity=10,
                                  COLORS=COLORS)
    
    try:
        mf_data = load_dict(filename.replace('ntwk', 'mf').replace('.h5', '.npz'))
        if 't' not in mf_data:
            mf_data['t'] = np.linspace(0, data['tstop'], mf_data['Vm'].shape[1])
        for i, label in enumerate(REC_POPS):
            AX[-1].plot(mf_data['t'], mf_data['X'][i,:], '--', lw=0.5, color=COLORS[i])
            AX[2+i].plot(mf_data['t'], 1e3*mf_data['Vm'][i,:], 'k-', lw=1, label='mean-field')
        # then Vm vs U-model vs MF
        #AX[2].plot(mf_data['t'], mf_data['desired_Vm'], 'k--', lw=2, label='U-model')
        AX[2].legend(frameon=False, loc='best')
    except FileNotFoundError:
        pass
        
    figM, _, _ = plot_matrix(data, REC_POPS, AFF_POPS, ge)
    
    ntwk.show()
    # ge.show()
    

def plot_matrix(Model, REC_POPS, AFF_POPS):


    pconnMatrix = np.zeros((len(REC_POPS)+len(AFF_POPS), len(REC_POPS)))
    
    for i, source_pop in enumerate(REC_POPS+AFF_POPS):
        for j, target_pop in enumerate(REC_POPS):
            pconnMatrix[i,j] = Model['p_%s_%s' % (source_pop, target_pop)]
            print('Model["p_%s_%s"] = %.4f' % (source_pop, target_pop, Model['p_%s_%s' % (source_pop, target_pop)]))
    
    fig, ax, acb = ge.figure(figsize=(1.,1.),
                             with_bar_legend=True)
    
    Lims = [np.round(100*pconnMatrix.min(),1)-.1,np.round(100*pconnMatrix.max(),1)+.1]
    
    ge.matrix(100*pconnMatrix.T, origin='upper', ax=ax, vmin=Lims[0], vmax=Lims[1])


    n, m = len(REC_POPS)+len(AFF_POPS), len(REC_POPS)
    for i, label in enumerate(REC_POPS+AFF_POPS):
        ge.annotate(ax, label, (-0.2, .9-i/n),\
            ha='right', va='center', color=COLORS[i])
    for i, label in enumerate(REC_POPS):
        ge.annotate(ax, label, (i/m+.25, -0.1),\
                    ha='right', va='top', color=COLORS[i], rotation=65)
    
    ge.build_bar_legend_continuous(acb, ge.viridis, bounds=[Lims[0], Lims[1]],
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

    plot_sim(sys.argv[-1], ge)
    
    # try:
    #     plot_sim(sys.argv[-1])
    # except BaseException:
    #     pass
    
