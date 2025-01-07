import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# this path determines where downloaded data will be stored
manifest_path = os.path.join("example_ecephys_project_cache", "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# only use brain_observatory_1.1 datasets
sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
brain_observatory_type_sessions.tail()
session_ids = brain_observatory_type_sessions['session_type'].index

import glob
#areas = vis_units['ecephys_structure_acronym'].values

def get_spikes(session, vis_units, drifting=False):
    if drifting:
        stim_pres = session.get_stimulus_table("drifting_gratings")
    else:
        stim_pres = session.get_stimulus_table("static_gratings")
        
    # using stim_pres
    spikes = session.presentationwise_spike_times(
        stimulus_presentation_ids = stim_pres.index.values,
        unit_ids = vis_units.index.values[:]
    )

    spikes["count"] = np.zeros(spikes.shape[0])
    spikes = spikes.groupby(["stimulus_presentation_id", "unit_id"]).count()
    sresp = pd.pivot_table(
        spikes, 
        values="count",
        index="stimulus_presentation_id", 
        columns="unit_id", 
        fill_value=0.0,
        aggfunc=np.sum
    )
    
    stim_pres = stim_pres[np.isin(stim_pres.index, sresp.index)]
    stims = stim_pres.orientation.values!='null'
    sresp = sresp.to_numpy().T
    sresp = sresp[:, stims]
    if drifting:
        istims = stim_pres.values[:, [1,-3]]
        istims = istims[stims].astype(np.float32)
    else:
        istims = stim_pres.values[:, [1,2,5]]
        istims = istims[stims].astype(np.float32)
    return sresp, istims

def tuning_repeats(sresp, istims, drifting=False):
    _,istim = np.unique(istims, axis=0, return_inverse=True)
    NN = sresp.shape[0]
    nstim = istim.size // 2
    two_repeats = np.zeros((nstim, NN, 2), np.float32)
    tun = np.zeros((np.unique(istim).size, sresp.shape[0]), np.float32)
    ik = 0
    for k,iori in enumerate(np.unique(istims[:,0])):
        tun[k] = sresp[:, istims[:,0]==iori].astype(np.float32).mean(axis=-1)
        if drifting:
            ist = np.logical_and(istims[:,0]==iori, istims[:,1]==8)
        else:
            ist = np.logical_and(np.logical_and(istims[:,0]==iori, istims[:,1]==0.0), istims[:,2]==0.04)
        ink = (ist).sum() // 2
        two_repeats[ik:ik+ink,:,0] = sresp[:, ist][:, :ink].T
        two_repeats[ik:ik+ink,:,1] = sresp[:, ist][:, ink:2*ink].T
        ik += ink
    two_repeats = two_repeats[:ik]
    
    # compute signal variance
    A = two_repeats.copy()
    A = (A - A.mean(axis=0)) / A.std(axis=0) + 1e-3
    sigvar =(A[:,:,0] * A[:,:,1]).mean(axis=0)
    
    return tun, two_repeats, sigvar

sigvar = np.zeros((0,), np.float32)
for sids in session_ids[1:]:
    session_id = sids
    print(session_id)
    session = cache.get_session_data(session_id)
    #session.metadata
    vis_units = session.units[np.isin(session.units["ecephys_structure_acronym"], 
                                      ["VISp"])]# , "VISrl", "VISl", "VISam", "VISpm"])]

    # drifting
    #sresp, istims = get_spikes(session, vis_units, True)
    #tun, two_repeats, sigvar = tuning_repeats(sresp, istims, True)
    #print(sigvar.mean())
    
    # static
    sresp, istims = get_spikes(session, vis_units, False)
    tun, two_repeats, sv0 = tuning_repeats(sresp, istims, False)
    print(sv0.mean())
    sigvar = np.append(sigvar, sv0, axis=0)
    print(two_repeats.shape)

session_ids.shape

session = cache.get_session_data(session_ids[0])

np.unique(session.get_stimulus_table('static_gratings')['spatial_frequency'])

saveroot = '/Data/CodingVisualCortex/EphysResult'

np.save(os.path.join(saveroot, 'ephys_sigvar.npy'), {'sigvar': sigvar, 'twor_ex': two_repeats})

print(np.nanmean(sigvar))
plt.hist(sigvar, 100)
plt.show()

idx=isort[0]
plt.plot(np.unique(ori)[1:], tun[:,idx])
plt.scatter(ori[ori!=-1] + 3*np.random.rand(nstim), 
            design[ori!=-1,idx] + .3*np.random.rand(nstim), 
            s=3, marker='o',alpha=0.1)
plt.ylim([0,12])

import decoders
istim = ori[ori!=-1]*np.pi/180
nstim = istim.size
itest = np.random.rand(nstim)<.25
itrain = np.ones((nstim,), np.bool)
itrain[itest] = 0
apred, error, ypred, logL, SNR, theta_pref = decoders.independent_decoder(
    design[ori!=-1].T, istim, itrain, itest, nbase=5)

np.median(np.abs(error))*180/np.pi

plt.scatter(istim[itest], apred, s=1)

plt.hist(error)
plt.show()

print(SNR.mean())
for a in np.unique(areas):
    print(a, SNR[areas==a].mean())
plt.hist(SNR,100)
plt.show()

SNR.shape

np.unique(istim)

np.unique(design.to_numpy().flatten())

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

design_arr = design.values.astype(np.float32)[ori!=-1]
targets_arr = ori[ori!=-1].astype(np.float32)
labels = np.unique(ori)[1:]

accuracies = []
confusions = []

for train_indices, test_indices in KFold(n_splits=5).split(design_arr):
    
    clf = svm.SVC(gamma="scale", kernel="rbf")
    clf.fit(design_arr[train_indices], targets_arr[train_indices])
    
    test_targets = targets_arr[test_indices]
    test_predictions = clf.predict(design_arr[test_indices])
    
    accuracy = 1 - (np.count_nonzero(test_predictions - test_targets) / test_predictions.size)
    print(accuracy)
    
    accuracies.append(accuracy)
    confusions.append(confusion_matrix(test_targets, test_predictions, labels))

plt.plot(tun[:,::10])
plt.show()

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(design_arr[train_indices], targets_arr[train_indices])
ypred = neigh.predict(design_arr[test_indices])
print(neigh.score(design_arr[test_indices], targets_arr[test_indices]))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100).fit(design_arr[train_indices], targets_arr[train_indices])
ypred = model.predict(design_arr[test_indices])
print(model.score(design_arr[test_indices], targets_arr[test_indices]))

mean_confusion = np.mean(confusions, axis=0)

fig, ax = plt.subplots(figsize=(8, 8))

img = ax.imshow(mean_confusion)
fig.colorbar(img)

ax.set_ylabel("actual")
ax.set_xlabel("predicted")

plt.show()