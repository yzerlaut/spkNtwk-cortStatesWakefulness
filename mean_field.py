import sys, pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve()))

import neural_network_dynamics.main as ntwk

from model import Model, REC_POPS, AFF_POPS

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
    np.save('data/COEFFS_pyrExc.npy', data['Model']['COEFFS'])
    
    
elif sys.argv[-1]=='tf-sim-demo':

    Model['filename'] = 'data/tf_data.npy'
    Model['NRN_KEY'] = 'pyrExc' # we scan this population
    
    Model['tstop'] = 1200
    Model['N_SEED'] = 1 # seed repetition
    Model['SEED'] = 1 # seed
    
    Model['POP_STIM'] = ['pyrExc', 'recInh']
    Model['F_pyrExc'] = 10.
    Model['F_recInh'] = 2.

    data = ntwk.run_single_cell_sim(Model, with_synaptic_currents=True, with_Vm=1, tdiscard=100)
    fig, AX = ntwk.plot_single_cell_sim(data)
    AX[1].annotate('$\\nu_{e}$=%.1fHz, $N_e$=%i, $p_{ee}$=%.2f  --> $\\nu_{e}^{tot}$=%.1fHz' %\
                   (Model['F_pyrExc'], Model['N_pyrExc'], Model['p_pyrExc_pyrExc'],
                    Model['F_pyrExc']*Model['N_pyrExc']*Model['p_pyrExc_pyrExc']),
                    (0.,1), xycoords='axes fraction')
    AX[1].annotate('$\\nu_{i}$=%.1fHz, $N_i$=%i, $p_{ie}$=%.2f  --> $\\nu_{i}^{tot}$=%.1fHz' %\
                   (Model['F_recInh'], Model['N_recInh'], Model['p_recInh_pyrExc'],
                    Model['F_recInh']*Model['N_recInh']*Model['p_recInh_pyrExc']),
                    (0.,0.), xycoords='axes fraction')
    AX[2].annotate('$\\nu_{out}$=%.1fHz' % data['fout_mean'],
                    (0.9,.9), xycoords='axes fraction', ha='center', va='top')
    fig.savefig('figures/tf_demo_sim.svg')
    ntwk.show()
    
elif sys.argv[-1]=='tf-plot':
    
    data = np.load('data/tf_data.npy', allow_pickle=True).item()

    data['Fout_mean'][data['Fout_mean']<=1e-2]=1e-2
    data['Fout_std'][data['Fout_mean']<=1e-2]=0

    from datavyz.main import graph_env
    ge = graph_env('manuscript')
    fig, ax, acb = ge.figure(figsize=(1.5,1.2),
                             bar_inset_loc=[1.1, 0., .05, 1.],
                             right=5, bottom=0.5, top=.4)

    try:
        data['Model']['COEFFS'] = np.load('data/COEFFS_pyrExc.npy')
        with_theory=True
    except FileNotFoundError:
        with_theory=False
        pass


    # translation into total excitatory and inhibitory frequencies
    data['F_recInh'] *= data['Model']['N_recInh']*data['Model']['p_recInh_pyrExc']
    data['F_pyrExc'] *= data['Model']['N_pyrExc']*data['Model']['p_pyrExc_pyrExc']
    data['Model']['N_recInh'] = 1
    data['Model']['p_recInh_pyrExc'] = 1.
    data['Model']['N_pyrExc'] = 1
    data['Model']['p_pyrExc_pyrExc'] = 1
    
    ntwk.make_tf_plot_2_variables(data,
                                  with_theory=with_theory,
                                  xkey='F_pyrExc', ckey='F_recInh',
                                  ylim=[1e-2, 100], yticks=[0.01, 0.1, 1, 10, 100],
                                  yticks_labels=['<0.01', '0.1', '1', '10', '100'],
                                  ylabel='$\\nu_{out}$ (Hz)',
                                  xlim=[70, 3000],
                                  xticks=[100, 200, 1000, 2000],
                                  xticks_labels=['100', '200', '1000', '2000'],
                                  xlabel='$\\nu_{e}^{tot}$ (Hz)',
                                  ckey_label='$\\nu_{i}^{tot}$ (Hz)',
                                  ge=ge, ax=ax, acb=acb)
    # fig.savefig('figures/tf_final_plot.svg')


    # from datavyz.plot_export import put_list_of_figs_to_svg_fig
    # put_list_of_figs_to_svg_fig(['figures/tf_demo_sim.svg',
    #                              'figures/tf_final_plot.svg'],
    #                             fig_name="figures/tf.svg")
    
    ntwk.show()
    
    
else:

    from datavyz.main import graph_env
    ge = graph_env()

    tstop, dt = 1, 5e-4

    DYN_SYSTEM = {}
    for pop in REC_POPS:
        DYN_SYSTEM[pop] = {'aff_pops':AFF_POPS,
                           'aff_pops_input_values':[3., 1.],
                           'x0':0.}
        Model['COEFFS_%s' % pop] = np.load('data/COEFFS_pyrExc.npy')
        
    # DYN_SYSTEM = {
    #     'Exc': {'aff_pops':AFF_POPS, 'aff_pops_input_values':[3.], 'x0':1.},
    #     'RecInh': {'aff_pops':['AffExc'], 'aff_pops_input_values':[3.], 'x0':1.}
    # }
    # INPUTS = {
    #     'AffExc_RecExc':np.ones(int(tstop/dt))*3.,
    #     'AffExc_RecInh':np.ones(int(tstop/dt))*3.
    #     }
        
    tstop, dt = 2, 5e-4
    t = np.arange(int(tstop/dt))*dt

    # DYN_SYSTEM = {
    #     'Exc': {'aff_pops':AFF_POPS, 'x0':1.},
    #     'oscillExc': {'aff_pops':AFF_POPS, 'x0':1.},
    #     'PvInh': {'aff_pops':AFF_POPS, 'x0':1.},
    #     'SstInh': {'aff_pops':AFF_POPS, 'x0':1.},
    #     'VipInh': {'aff_pops':AFF_POPS, 'x0':1.}}
    INPUTS = {}
    for aff in AFF_POPS:
        for pop in REC_POPS:
            INPUTS['%s_%s' % (aff, pop)] = 0.5*np.ones(len(t))

    CURRENT_INPUTS = {'oscillExc':190*(1+np.sin(2*np.pi*3.*t))} # very low current just for demo
    
    X0 = ntwk.mean_field.find_fp(Model,
                            DYN_SYSTEM,
                            INPUTS=INPUTS,
                            CURRENT_INPUTS=CURRENT_INPUTS,
                            dt=dt,
                 tstop=tstop,
                 replace_x0=True)
    
    X = ntwk.mean_field.solve_mean_field_first_order(Model, DYN_SYSTEM,
                                     INPUTS=INPUTS,
                                     CURRENT_INPUTS=CURRENT_INPUTS,
                                     dt=dt, tstop=tstop)

    # print(mean_field.get_full_statistical_quantities(Model, DYN_SYSTEM))
    
    fig, ax = ge.figure()
    ge.plot(Y=[X[pop] for pop in REC_POPS],
            # axes_args=dict(yticks=[0.1,1.,10], ylim=[1e-3, 100]),
            COLORS=[ge.g, ge.b, ge.r, ge.purple, ge.orange], ax=ax)
    ge.show()

