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
# # Decoding stimulus identity from neural responses
#
# We implement here a nearest-neighbor decoder of neural activity.

# %%
# general python modules
import sys, os, pprint
import numpy as np
import matplotlib.pylab as plt

# *_-= root =-_*
root_folder = os.path.join(os.path.expanduser('~'), 'work', 'spkNtwk-cortStatesWakefulness') # UPDATE to your folder location
# -- physion core
sys.path.append(os.path.join(root_folder, 'physion'))
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.dataviz.show_data import MultimodalData, EpisodeResponse
# -- physion data visualization
sys.path.append(os.path.join(root_folder, 'datavyz'))
from datavyz import ge

# %% [markdown]
# ## Loading data and preprocessing

# %%
root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'curated')
filename = '2022_06_13-14-17-33.nwb'
root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA', '6juil')
filename = '2022_07_06-14-31-07.nwb'
data = Data(os.path.join(root_datafolder, filename))
print(data.protocols)

# %% [markdown]
# ### Natural Images episodes

# %%
episodes_NI = EpisodeResponse(os.path.join(root_datafolder, filename), 
                              protocol_id=1,#data.get_protocol_id('NI-VSE-3images-2vse-30trials'),
                              quantities=['dFoF', 'Pupil', 'Running-Speed'],
                              dt_sampling=30, # ms, to avoid to consume to much memory
                              verbose=True, prestim_duration=1.5)

print('episodes_NI:', episodes_NI.protocol_name)
print('varied parameters:', episodes_NI.varied_parameters)

# %% [markdown]
# ### Gaussian Blobs episodes

# %%
episodes_GB = EpisodeResponse(os.path.join(root_datafolder, filename), 
                              quantities=['dFoF', 'Pupil', 'Running-Speed'],
                              protocol_id=0,#data.get_protocol_id('gaussian-blobs'),
                              dt_sampling=30, # ms, to avoid to consume to much memory
                              verbose=True, prestim_duration=1.5)

print('episodes_GB:', episodes_GB.protocol_name)
print('varied parameters:', episodes_GB.varied_parameters)


# %% [markdown]
# ## Visualizing evoked responses

# %%
def find_responsive_rois(episodes, stim_key,
                         value_threshold=0.5,
                         significance_threshold=0.01):
    
    responsive_rois = [[] for i in range(len(episodes.varied_parameters[stim_key]))]
    
    duration = episodes.data.metadata['Protocol-%i-presentation-duration' % (episodes.protocol_id+1)]

    # looping over neurons
    for roi in range(episodes.data.nROIs):
        roi_summary = episodes.compute_summary_data(dict(interval_pre=[-2,0], 
                                                         interval_post=[0,duration],
                                                         test='wilcoxon', 
                                                         positive=True),
                                                     response_args={'quantity':'dFoF', 
                                                                    'roiIndex':roi},
                                                   response_significance_threshold=significance_threshold)
        for istim, stim in enumerate(episodes.varied_parameters[stim_key]):
            if roi_summary['significant'][istim] and (roi_summary['value'][istim]>value_threshold):
                responsive_rois[istim].append(roi)
                
    return responsive_rois


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

show_stim_evoked_resp(episodes_NI, 'Image-ID', 0);

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
