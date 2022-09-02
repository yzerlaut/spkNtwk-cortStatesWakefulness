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
                                      with_visual_stim=True,
                                      verbose=False)
print(data.protocols)

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
                         stim_indices,
                         ROI_SUMMARIES,
                         value_threshold=0.5,
                         significance_threshold=0.01):
    

    cond = np.ones(len(ROI_SUMMARIES[0]['value']), dtype=bool)
    for key, index in zip(stim_keys, stim_indices):
        cond = (cond & (ROI_SUMMARIES[0][key+'-index']==index))
    
    responsive_rois = []
    
    
    # looping over neurons
    for roi in range(episodes.dFoF.shape[0]):
        
        if ROI_SUMMARIES[roi]['significant'][cond] and ROI_SUMMARIES[roi]['relative_value'][cond]>value_threshold:
            responsive_rois.append(roi)
                
    return responsive_rois

# %%
responsive_rois = find_responsive_rois(episodes_NI, ['Image-ID', 'VSE-seed'], [0,0], ROI_SUMMARIES)
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
stim_keys, stim_indices = ['Image-ID', 'VSE-seed'], [0,0]
responsive_rois = find_responsive_rois(episodes_NI, stim_keys, stim_indices, ROI_SUMMARIES)
show_stim_evoked_resp(Episodes_NI, stim_keys, stim_indices,
                      responsive_rois=np.random.choice(responsive_rois,8,replace=False));

# %%
stim_keys, stim_indices = ['Image-ID', 'VSE-seed'], [1,0]
responsive_rois = find_responsive_rois(episodes_NI, stim_keys, stim_indices, 
                                      ROI_SUMMARIES)

fig, AX = physion.analysis.behavioral_modulation.plot_resp_dependency(Episodes_NI,
                                                            stim_keys, stim_indices,
                                                            responsive_rois,
                                                            running_threshold=0.2,
                                                            N_selected=20, selection_seed=20)


# %%
import itertools

running_threshold = 0.2 # cm/s

stim_keys = [key for key in Episodes_NI.varied_parameters if key!='repeat']
set_of_stim_indices = list(itertools.product(*[range(len(Episodes_NI.varied_parameters[key])) for key in stim_keys]))

for stim_indices in set_of_stim_indices:
    
    responsive_rois = find_responsive_rois(Episodes_NI, stim_keys, stim_indices, 
                                           ROI_SUMMARIES)

    fig, AX = physion.analysis.behavioral_modulation.plot_resp_dependency(Episodes_NI,
                                                            stim_keys, stim_indices,
                                                            responsive_rois,
                                                            running_threshold=running_threshold,
                                                            N_selected=20, selection_seed=20)


# %% [markdown]
# ## Nearest-neighbor classifier for the classification of neural patterns

# %%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def run_model_NN_single_trial(episodes, 
                              stim_keys, set_of_stim_indices,
                              episode_cond=None,
                              seed=1, test_size=0.5):
    
    # -- assign a single label to each stimulus type
    labels = -1*np.ones(episodes.dFoF.shape[0]) # a value for each episode
    for label, stim_indices in enumerate(set_of_stim_indices):
        # get the condition corresponding to those joint stim keys
        label_cond = np.ones(episodes.dFoF.shape[0], dtype=bool)
        for key, index in zip(stim_keys, stim_indices):
            label_cond = label_cond & (getattr(episodes, key)==episodes.varied_parameters[key][index])
        # give it the right label
        labels[label_cond] = label
    
    # -- filter episodes if needed (e.g. run only)
    if episode_cond is None:
        ep_ids = np.arange(episodes.dFoF.shape[0]) # all episodes
    else:
        ep_ids = np.arange(episodes.dFoF.shape[0])[episode_cond]
        
    X_train, X_test, y_train, y_test = train_test_split([episodes.dFoF[i,:,:].flatten() for i in ep_ids],
                                                        labels[ep_ids],
                                                        test_size=test_size, random_state=seed)

    nn_model = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(X_train, y_train)
    y_predicted = nn_model.predict(X_test)
    
    return np.sum((y_predicted-y_test)==0)/len(y_test)


def run_model_NN_trial_average(episodes, 
                               stim_keys, set_of_stim_indices,
                               episode_cond=None,
                               seed=1, test_size=0.5):
    
    # -- assign a single label to each stimulus type
    labels = -1*np.ones(episodes.dFoF.shape[0]) # a value for each episode
    for label, stim_indices in enumerate(set_of_stim_indices):
        # get the condition corresponding to those joint stim keys
        label_cond = np.ones(episodes.dFoF.shape[0], dtype=bool)
        for key, index in zip(stim_keys, stim_indices):
            label_cond = label_cond & (getattr(episodes, key)==episodes.varied_parameters[key][index])
        # give it the right label
        labels[label_cond] = label
    
    # -- filter episodes if needed (e.g. run only)
    if episode_cond is None:
        ep_ids = np.arange(episodes.dFoF.shape[0]) # all episodes
    else:
        ep_ids = np.arange(episodes.dFoF.shape[0])[episode_cond]
        
    X_train, X_test, y_train, y_test = train_test_split([episodes.dFoF[i,:,:].flatten() for i in ep_ids],
                                                        labels[ep_ids],
                                                        test_size=test_size, random_state=seed)

    X_train_average, y_train_average = [], []
    
    for i in range(len(set_of_stim_indices)):
        stim_cond = np.flatnonzero(y_train==i)
        if len(stim_cond)>0:
            X_train_average.append(np.stack([X_train[l] for l in stim_cond], axis=-1).mean(axis=-1))
            y_train_average.append(i)
        
    nn_model = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(X_train_average, y_train_average)
    y_predicted = nn_model.predict(X_test)
    
    return np.sum((y_predicted-y_test)==0)/len(y_test)

run_model_NN_trial_average(Episodes_NI, stim_keys, set_of_stim_indices)

# %%
running_cond = Episodes_NI.running_speed.mean(axis=-1)>running_threshold
MEANS, STDS = [], []
for i, cond, color in zip(range(3),
                       [None, ~running_cond, running_cond],
                       [ge.grey, ge.blue, ge.orange]):
    
    accuracies = [run_model_NN_single_trial(Episodes_NI,
                                            stim_keys, set_of_stim_indices,
                                            episode_cond=cond,
                                            seed=i) for i in range(10)]
    
    MEANS.append(100.*np.mean(accuracies))
    STDS.append(100.*np.std(accuracies))

# %%
chance = 100./len(set_of_stim_indices)
fig, ax = ge.bar(np.array(MEANS)-chance,sy=STDS, COLORS=[ge.grey, ge.blue, ge.orange])
ge.set_plot(ax, ylabel='decoding\naccuracy (%)', 
            xticks=range(3), xticks_labels=['all', 'still', 'run'],
            yticks=[0, .5*(100-chance), 100-chance],
            ylim=[0, 100-chance],
            yticks_labels=['chance', '%.1f' % (.5*(100+chance)), '100'],
            title='NN single-trial')
ax.plot([-1,4], np.zeros(2), 'k--', lw=1)

# %%
running_cond = Episodes_NI.running_speed.mean(axis=-1)>running_threshold
MEANS, STDS = [], []
for i, cond, color in zip(range(3),
                       [None, ~running_cond, running_cond],
                       [ge.grey, ge.blue, ge.orange]):
    
    accuracies = [run_model_NN_trial_average(Episodes_NI,
                                            stim_keys, set_of_stim_indices,
                                            episode_cond=cond,
                                            seed=i) for i in range(10)]
    
    MEANS.append(100.*np.mean(accuracies))
    STDS.append(100.*np.std(accuracies))

# %%
chance = 100./len(set_of_stim_indices)
fig, ax = ge.bar(np.array(MEANS)-chance,sy=STDS, COLORS=[ge.grey, ge.blue, ge.orange])
ge.set_plot(ax, ylabel='decoding\naccuracy (%)', 
            xticks=range(3), xticks_labels=['all', 'still', 'run'],
            yticks=[0, .5*(100-chance), 100-chance],
            ylim=[0, 100-chance],
            yticks_labels=['chance', '%.1f' % (.5*(100+chance)), '100'],
            title='NN trial-average')
ax.plot([-1,4], np.zeros(2), 'k--', lw=1)
