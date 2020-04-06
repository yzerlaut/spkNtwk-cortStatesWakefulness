import sys, pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve()))

import neural_network_dynamics.main as ntwk

from model import Model, REC_POPS, AFF_POPS

# by waiting for the proper params scan, let's pick those of the Cell Rep study
for pop in REC_POPS:
    Model['COEFFS_%s' % pop] = np.load('neural_network_dynamics/configs/The_Spectrum_of_Asynch_Dyn_2018/COEFFS_RecExc.npy')


if sys.argv[-1]=='tf-sim':
    
    Model['filename'] = 'data/tf_data.npy'
    Model['NRN_KEY'] = 'pyrExc' # we scan this population
    
    Model['tstop'] = 30000
    
    Model['N_SEED'] = 4 # seed repetition
    
    Model['POP_STIM'] = ['pyrExc', 'recInh']
    
    Model['F_pyrExc_array'] = np.logspace(np.log10(0.05), np.log10(50), 70)
    Model['F_recInh_array'] = np.logspace(np.log10(0.05), np.log10(50), 50)
    
    ntwk.generate_transfer_function(Model)
    
    print('Results of the simulation are stored as:', 'data/tf_data.npy')

elif sys.argv[-1]=='tf-fit':
    
    data = np.load('data/tf_data.npy', allow_pickle=True).item()

    # from neural_network_dynamics.theory.fitting_tf import fit_data
    data['Model']['COEFFS'] = ntwk.fit_tf_data(data, order=2)
    np.save('data/COEFFS_neuron.npy', data['Model']['COEFFS'])
    
    
elif sys.argv[-1]=='tf-plot':
    
    data = np.load('data/tf_data.npy', allow_pickle=True).item()

    data['Fout_mean'][data['Fout_mean']<=1e-2]=1e-2
    data['Fout_std'][data['Fout_mean']<=1e-2]=0

    from datavyz.main import graph_env
    ge = graph_env('manuscript')
    fig, ax, acb = ge.figure(figsize=(1.5,1.2), bar_inset_loc=[1.1, 0., .05, 1.],
                             right=5, bottom=0.8)
    
    ntwk.make_tf_plot_2_variables(data,
                                  ge=ge, ax=ax, acb=acb,
                                  xkey='F_pyrExc', ckey='F_recInh',
                                  ylim=[1e-2, 80], yticks=[0.01, 0.1, 1, 10], yticks_labels=['<0.01', '0.1', '1', '10'], ylabel='$\\nu_{out}$ (Hz)',
                                  xlim=[2, 100], xticks=[2, 5, 10, 20, 50], xticks_labels=['2', '5', '10', '20', '50'], xlabel='$\\nu_{e}$ (Hz)',
                                  fig_args={'figsize':(1.5,1.5), 'with_space_for_bar_legend':True})
    ntwk.show()
    
    
else:
    
    from graphs.my_graph import graphs
    mg = graphs()

    tstop, dt = 1, 5e-4

    DYN_SYSTEM = {
        'Exc': {'aff_pops':AFF_POPS, 'aff_pops_input_values':[3.], 'x0':1.},
        'RecInh': {'aff_pops':['AffExc'], 'aff_pops_input_values':[3.], 'x0':1.}
    }
    INPUTS = {
        'AffExc_RecExc':np.ones(int(tstop/dt))*3.,
        'AffExc_RecInh':np.ones(int(tstop/dt))*3.
        }
        
    tstop, dt = 2, 5e-4
    t = np.arange(int(tstop/dt))*dt

    DYN_SYSTEM = {
        'Exc': {'aff_pops':AFF_POPS, 'x0':1.},
        'oscillExc': {'aff_pops':AFF_POPS, 'x0':1.},
        'PvInh': {'aff_pops':AFF_POPS, 'x0':1.},
        'SstInh': {'aff_pops':AFF_POPS, 'x0':1.},
        'VipInh': {'aff_pops':AFF_POPS, 'x0':1.}}
    INPUTS = {}
    for aff in AFF_POPS:
        for pop in REC_POPS:
            INPUTS['%s_%s' % (aff, pop)] = 0.5*np.ones(len(t))

    CURRENT_INPUTS = {'oscillExc':190*(1+np.sin(2*np.pi*3.*t))} # very low current just for demo
    
    X0 = mean_field.find_fp(Model,
                            DYN_SYSTEM,
                            INPUTS=INPUTS,
                            CURRENT_INPUTS=CURRENT_INPUTS,
                            dt=dt,
                 tstop=tstop,
                 replace_x0=True)
    
    X = mean_field.solve_mean_field_first_order(Model, DYN_SYSTEM,
                                     INPUTS=INPUTS,
                                     CURRENT_INPUTS=CURRENT_INPUTS,
                                     dt=dt, tstop=tstop)

    # print(mean_field.get_full_statistical_quantities(Model, DYN_SYSTEM))
    
    fig, ax = mg.figure()
    ax.set_yscale('log')
    mg.plot(Y=[X['Exc'], X['oscillExc'], X['PvInh'], X['SstInh'], X['VipInh']],
            COLORS=[mg.g, mg.b, mg.r, mg.purple, mg.orange], ax=ax, axes_args=dict(yticks=[0.1,1.,10], ylim=[1e-3, 100]))
    mg.show()

