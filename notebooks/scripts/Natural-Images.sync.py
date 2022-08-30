# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Response to Natural Images

# %%
# general python modules
import sys, os, pprint
import numpy as np
import matplotlib.pylab as plt
from importlib import reload # to be able to call "reload(physion)"

# *_-= root =-_*
physion_folder = os.path.join(os.path.expanduser('~'), 'work', 'spkNtwk-cortStatesWakefulness', 'physion') # UPDATE
# -- import physion core:
sys.path.append(physion_folder)
import physion
# -- import data visualization module:
sys.path.append(os.path.join(physion_folder, 'dataviz', 'datavyz'))
from datavyz import graph_env
ge = graph_env('notebook')

# %% [markdown]
# ## Loading data and preprocessing

# %%
root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'curated')
filename = '2022_06_13-14-17-33.nwb'
root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA', '6juil')
filename = '2022_07_06-14-31-07.nwb'
data = physion.analysis.read_NWB.Data(os.path.join(root_datafolder, filename),
                                      with_visual_stim=True)
print(data.protocols)

# %%
#ge.image(data.visual_stim.get_vse(120))
os.path.join(root_datafolder, filename)

# %%
data.init_visual_stim()
data.visual_stim

# %% [markdown]
# ### Natural Images episodes

# %%
data = physion.analysis.read_NWB.Data(os.path.join(root_datafolder, filename))

episodes_NI = physion.analysis.process_NWB.EpisodeResponse(data,
                                      protocol_id=1,#data.get_protocol_id('NI-VSE-3images-2vse-30trials'),
                                      quantities=['dFoF', 'Pupil', 'Running-Speed'],
                                      dt_sampling=30, # ms, to avoid to consume to much memory
                                        
                                                           
                                      verbose=True, prestim_duration=1.5)

print('episodes_NI:', episodes_NI.protocol_name)
print('varied parameters:', episodes_NI.varied_parameters)

# %%
reload(physion)
Episodes_NI = physion.dataviz.show_data.EpisodeResponse(episodes_NI)

print('episodes_NI:', Episodes_NI.protocol_name)
print('varied parameters:', Episodes_NI.varied_parameters)

# %% [markdown]
# ## Visualizing evoked responses

# %%
episodes_NI.visual_stim.protocol

# %%
ROI_SUMMARIES = [episodes_NI.compute_summary_data(dict(interval_pre=[-episodes_NI.visual_stim.protocol['presentation-interstim-period'],0], 
                                                       interval_post=[0,episodes_NI.visual_stim.protocol['presentation-duration']],
                                                       test='wilcoxon', 
                                                       positive=True),
                                                     response_args={'quantity':'dFoF', 
                                                                    'roiIndex':roi},
                                                   response_significance_threshold=0.01) for roi in range(episodes_NI.dFoF.shape[0])]

# %%
ROI_SUMMARIES = [episodes_NI.compute_summary_data(dict(interval_pre=[-episodes_NI.visual_stim.protocol['presentation-interstim-period'],0],
                                                    interval_post=[0,episodes_NI.visual_stim.protocol['presentation-duration']],
                                                    test='wilcoxon', 
                                                    positive=True),
                                                response_args={'quantity':'dFoF', 
                                                               'roiIndex':roi},
                                                response_significance_threshold=0.01) for roi in range(episodes_NI.dFoF.shape[0])]
def find_responsive_rois(episodes, 
                         stim_keys,
                         stim_values,
                         ROI_SUMMARIES,
                         value_threshold=0.5,
                         significance_threshold=0.01):
    

    cond = np.ones(len(ROI_SUMMARIES[0]['value']), dtype=bool)
    for key, val in zip(stim_keys, stim_values):
        cond = (cond & (ROI_SUMMARIES[0][key]==val))
    
    responsive_rois = []
    
    
    # looping over neurons
    for roi in range(episodes.dFoF.shape[0]):
        
        if ROI_SUMMARIES[roi]['significant'][cond] and ROI_SUMMARIES[roi]['relative_value'][cond]>value_threshold:
            responsive_rois.append(roi)
                
    return responsive_rois

# %%
responsive_rois = find_responsive_rois(episodes_NI, ['Image-ID', 'VSE-seed'], [2,8], ROI_SUMMARIES)
print(responsive_rois)

# %%
# let's pick a specific stim
stim_index = 1

def show_stim_evoked_resp(episodes, stim_key, stim_index,
                          quantity='dFoF',
                          responsive_rois = None,
                          value_threshold=0.5,
                          with_stim_inset=True,
                          Nrois=10):
    
    if responsive_rois is None:
        responsive_rois = np.arange(getattr(episodes, quantity).shape[1])
        
    return episodes.plot_evoked_pattern(episodes.find_episode_cond(stim_key, stim_index),
                                        rois=np.random.choice(responsive_rois, 
                                                         min([Nrois, len(responsive_rois)]), replace=False),
                                        quantity=quantity, with_stim_inset=with_stim_inset)


# %%
stim_keys, stim_indices = ['Image-ID', 'VSE-seed'], [0,1]
responsive_rois = find_responsive_rois(episodes_NI, stim_keys, 
                                      [episodes_NI.varied_parameters[k][i] for k,i in zip(stim_keys, stim_indices)],
                                      ROI_SUMMARIES)
show_stim_evoked_resp(Episodes_NI, stim_keys, stim_indices,
                      responsive_rois=np.random.choice(responsive_rois,8,replace=False));


# %%
def plot_resp_dependency(Episodes,
                         stim_keys=['Image-ID', 'VSE-seed'],
                         stim_indices=[0,0],
                         running_threshold=0.1,
                         selected_rois=None,
                         raster_norm='', traces_norm=''):
    
    all_eps = Episodes.find_episode_cond(stim_keys, stim_indices)
    
    if selected_rois is None:
        selected_rois = np.random.choice(np.arange(Episodes.dFoF.shape[0]), 8)
    
    
    fig, AX = ge.figure(axes_extents=[[[1,3] for i in range(4)],                                                         
                                     [[1,6] for i in range(4)]],                                 
                        figsize=(1,.2), left=0.5, right=2, wspace=0.4)                                                                       

    
    ge.bar_legend(AX[0][0],
                  colorbar_inset=dict(rect=[-0.5,.2,.03,.6], facecolor=None),
                  colormap=ge.binary,
                  bar_legend_args={},
                  #label='n. $\Delta$F/F', bounds=None, ticks = None, ticks_labels=None, no_ticks=False,
                  label='$\Delta$F/F', bounds=[0,2], ticks = [0,1,2], ticks_labels=['0','1','>2'], no_ticks=False,
                  orientation='vertical')

    
    ge.bar_legend(AX[1][0],
                  colorbar_inset=dict(rect=[-0.5,.2,.02,.6], facecolor=None),
                  colormap=ge.jet,
                  bar_legend_args={},
                  label='single trials', bounds=None, ticks = None, ticks_labels=None, no_ticks=False,
                  orientation='vertical')
    
    
    stim_inset = ge.inset(fig, [0.9,0.85,0.1,0.15])
    Episodes_NI.visual_stim.plot_stim_picture(np.flatnonzero(all_eps)[0],
                                              ax=stim_inset, vse=True)
    
    behav_inset = ge.inset(fig, [0.76,0.8,0.1,0.2])
    Episodes_NI.behavior_variability(episode_condition=all_eps,
                                     threshold2=running_threshold, ax=behav_inset)

    
    running = all_eps & (Episodes.running_speed.mean(axis=1)>running_threshold)
    still = all_eps & (Episodes.running_speed.mean(axis=1)<=running_threshold)


    #norm_ROIS = [(np.inf,-np.inf) for r in range(selected_rois)]
    scale_ROIS = np.ones(len(selected_rois))
    
    for cond, axP, axT, label, color in zip([all_eps, running, still], AX[0], AX[1],
                    ['all eps', 'running', 'still'], ['k', ge.blue, ge.orange]):
        
        ge.title(axP, '%s (n=%i)' % (label, np.sum(cond)), color=color)
        
        mean_resp = Episodes_NI.dFoF[cond,:,:].mean(axis=0)

        axP.imshow(mean_resp,           
                   cmap=ge.binary,
                   aspect='auto', interpolation='none',                                                                
                   vmin=0, vmax=2,                                                                                     
                   origin='lower',
                   extent = (Episodes_NI.t[0], Episodes_NI.t[-1],                                                                    
                             0, mean_resp.shape[1]))
        min_dFoF_range = 2.
        for ir, r in enumerate(selected_rois):
            roi_resp = Episodes_NI.dFoF[cond, r, :]
            scale = max([min_dFoF_range, np.max(roi_resp-roi_resp.mean())]) # 2 dFoF is the min scale range
            # plotting eps with that scale
            for iep in range(np.sum(cond)):
                axT.plot(Episodes.t, ir+(roi_resp[iep,:]-roi_resp.mean())/scale,
                         color=ge.jet(iep/(np.sum(cond)-1)), lw=.5)
            # plotting scale
            axT.plot([Episodes.t[-1], Episodes.t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=1.5)
                
            if 'all' in label:
                ge.annotate(axT, 'roi#%i ' % (r+1), (Episodes.t[0], ir), xycoords='data',
                            ha='right', size='xx-small')
                scale_ROIS[ir] = scale
               
            if label=='running':
                ge.plot(Episodes.t, 
                        ir+Episodes_NI.dFoF[cond, r, :].mean(axis=0)/scale_ROIS[ir],
                        sy=Episodes_NI.dFoF[cond, r, :].std(axis=0)/scale_ROIS[ir],
                        ax=AX[1][3], color=ge.blue, no_set=True)
                
            if label=='still':
                ge.plot(Episodes.t, 
                        ir+Episodes_NI.dFoF[cond, r, :].mean(axis=0)/scale_ROIS[ir],
                        sy=Episodes_NI.dFoF[cond, r, :].std(axis=0)/scale_ROIS[ir],
                        ax=AX[1][3], color=ge.orange, no_set=True)
                AX[1][3].plot([Episodes.t[-1], Episodes.t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=1.5)

        ge.annotate(axT, '1$\Delta$F/F', (Episodes.t[-1], 0), xycoords='data',
                    rotation=90, size='small')
        ge.set_plot(axT, [], xlim=[Episodes.t[0], Episodes.t[-1]])
        ge.draw_bar_scales(axT, Xbar=1, Xbar_label='1s', Ybar=1e-12)
        
    ge.set_plot(AX[1][3], [], xlim=[Episodes.t[0], Episodes.t[-1]])
    ge.draw_bar_scales(AX[1][3], Xbar=1, Xbar_label='1s', Ybar=1e-12)
    
    # comparison
    AX[0][3].axis('off')

    vse_shifts = Episodes.visual_stim.vse['t'][Episodes.visual_stim.vse['t']<Episodes.visual_stim.protocol['presentation-duration']]
    Nbar=100
    for ax,ax1 in zip(AX[0][:3], AX[1][:3]):
        ge.set_plot(ax, [], xlim=[Episodes.t[0], Episodes.t[-1]])
        ax.plot([Episodes.t[0], Episodes.t[0]], [0, Nbar], 'k-', lw=2)
        ge.annotate(ax, '%i rois' % Nbar, (Episodes.t[0], 0), rotation=90, xycoords='data', ha='right')                   
        for t in [0]+list(vse_shifts):
            ax.plot(t*np.ones(2), ax.get_ylim(), 'r--', lw=0.3)
            ax1.plot(t*np.ones(2), ax1.get_ylim(), 'r--', lw=0.3)
    
stim_keys, stim_indices = ['Image-ID', 'VSE-seed'], [1,0]
responsive_rois = find_responsive_rois(episodes_NI, stim_keys, 
                                      [episodes_NI.varied_parameters[k][i] for k,i in zip(stim_keys, stim_indices)],
                                      ROI_SUMMARIES)
plot_resp_dependency(Episodes_NI, stim_keys, stim_indices,
                     running_threshold=0.2,
                     selected_rois=np.random.choice(responsive_rois,10,replace=False),
                     raster_norm='')

# %%
# ge.bar_legend?

# %%
len(episodes_NI.time_start), episodes_NI.protocol_cond_in_full_data.sum()

# %%
show_stim_evoked_resp(episodes_NI, 'Image-ID', 1);
show_stim_evoked_resp(episodes_NI, 'Image-ID', 2);

# %%
show_stim_evoked_resp(episodes_GB, 'contrast', 4, 
                      quantity='dFoF', with_stim_inset=False);

# %%
show_stim_evoked_resp(episodes_GB, 'center-time', 0, 
                      quantity='dFoF', with_stim_inset=False);
show_stim_evoked_resp(episodes_GB, 'center-time', 1, 
                      quantity='dFoF', with_stim_inset=False);
show_stim_evoked_resp(episodes_GB, 'center-time', 2, 
                      quantity='dFoF', with_stim_inset=False);

# %% [markdown]
# ## Nearest-neighbor classifier for the classification of neural patterns

# %%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def run_model(episodes, key, values, seed=1, test_size=0.5):
    X_train, X_test, y_train, y_test = train_test_split([episodes.dFoF[i,:,:].flatten() for i in range(episodes.dFoF.shape[0])],
                                                        values,
                                                        test_size=test_size, random_state=seed)
    #print(X_train.shape)
    nn_model = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(X_train, y_train)
    y_predicted = nn_model.predict(X_test)
    return np.sum((y_predicted-y_test)==0)/len(y_test)


# %%
accuracies_NI = [run_model(episodes_NI, 
                           'Image-ID', 
                           getattr(episodes_NI, 'Image-ID'),
                           seed=i) for i in range(30)]

# %%
accuracies_GB = [run_model(episodes_GB, 
                           'center-time', 
                           np.array(10*getattr(episodes_GB, 'center-time'), dtype=int), 
                           seed=i) for i in range(30)]

# %%
fig, ax = ge.bar([100*np.mean(accuracies_NI), 100*np.mean(accuracies_GB)],
                 sy=[100*np.std(accuracies_NI), 100*np.std(accuracies_GB)])
ge.set_plot(ax, ylabel='neurometric task\naccuracy (%)', xticks=[0,1], 
            xticks_labels=['NI-discrimination', 'Cue-detection'],
            title='all trials',
            yticks=[0,50,100],
            xticks_rotation=30)
ax.plot([-0.5,1.5], 1./6.*100*np.ones(2), 'k--', lw=0.5)
ge.annotate(ax, 'chance level', (1,0))

# %% [markdown]
# ## Behavior

# %%
fig, AX = ge.figure(axes=(2,1), figsize=(1.2,1.5), wspace=1.5)

threshold = 0.1 # cm/s

for ax, EPISODES, title in zip(AX, [episodes_NI, episodes_GB],
                               ['NI episodes', 'GB episodes']):
    running = np.mean(EPISODES.RunningSpeed, axis=1)>threshold

    ge.scatter(np.mean(EPISODES.pupilSize, axis=1)[running], 
               np.mean(EPISODES.RunningSpeed, axis=1)[running],
               ax=ax, no_set=True, color=ge.blue, ms=5)
    ge.scatter(np.mean(EPISODES.pupilSize, axis=1)[~running], 
               np.mean(EPISODES.RunningSpeed, axis=1)[~running],
               ax=ax, no_set=True, color=ge.orange, ms=5)
    ge.set_plot(ax, xlabel='pupil size (mm)', ylabel='run. speed (cm/s)', title=title)
    ax.plot(ax.get_xlim(), threshold*np.ones(2), 'k--', lw=0.5)
    ge.annotate(ax, 'n=%i' % np.sum(running), (0,1), va='top', color=ge.blue)
    ge.annotate(ax, '\nn=%i' % np.sum(~running), (0,1), va='top', color=ge.orange)

# %%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def run_model_w_cond(episodes, condition, key, values, seed=1, test_size=0.5):
    cond_array = np.arange(len(condition))[condition]
    X_train, X_test, y_train, y_test = train_test_split([episodes.dFoF[i,:,:].flatten() for i in cond_array],
                                                        values[condition],
                                                        test_size=test_size, random_state=seed)
    #print(X_train.shape)
    nn_model = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(X_train, y_train)
    y_predicted = nn_model.predict(X_test)
    return np.sum((y_predicted-y_test)==0)/len(y_test)


# %%

threshold = 0.1 # cm/s

# NI
running = np.mean(episodes_NI.RunningSpeed, axis=1)>threshold
accuracies_NI_run = [run_model_w_cond(episodes_NI, running,
                                      'Image-ID', 
                                      getattr(episodes_NI, 'Image-ID'),
                                      seed=i) for i in range(10)]
accuracies_NI_still = [run_model_w_cond(episodes_NI, ~running,
                                      'Image-ID', 
                                      getattr(episodes_NI, 'Image-ID'),
                                      seed=i) for i in range(10)]


# GB
running = np.mean(episodes_GB.RunningSpeed, axis=1)>threshold
accuracies_GB_run = [run_model_w_cond(episodes_GB, running,
                                      'center-time', 
                                      np.array(10*getattr(episodes_GB, 'center-time'), dtype=int), 
                                      seed=i) for i in range(10)]
accuracies_GB_still = [run_model_w_cond(episodes_GB, ~running,
                                      'center-time', 
                                      np.array(10*getattr(episodes_GB, 'center-time'), dtype=int), 
                                      seed=i) for i in range(10)]



# %%
fig, ax = ge.bar([100*np.mean(accuracies_NI_still), 100*np.mean(accuracies_NI_run), 0,
                  100*np.mean(accuracies_GB_still), 100*np.mean(accuracies_GB_run)],
                 sy=[100*np.std(accuracies_NI_still), 100*np.std(accuracies_NI_run), 0,
                     100*np.std(accuracies_GB_still), 100*np.std(accuracies_GB_run)],
                 COLORS=[ge.orange, ge.blue, 'k', ge.orange, ge.blue])

ge.set_plot(ax, ylabel='neurometric task\naccuracy (%)', xticks=[0,1,3,4], 
            xticks_labels=['NI-discrimination', '', '', 'Cue-detection'],
            yticks=[0,50,100],
            xticks_rotation=30)

ge.annotate(ax, 'still trials', (1,1), va='top', color=ge.orange, bold=True)
ge.annotate(ax, '\nrun trials', (1,1), va='top', color=ge.blue, bold=True)
ax.plot([-0.5,4.5], 1./6.*100*np.ones(2), 'k--', lw=0.5)


# %%
