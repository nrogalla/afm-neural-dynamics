import h5py
import numpy as np
import math
import os
import jax.numpy as jnp
import copy
import re

####
# Dataloading adapted from algonauts challenge testkit
####
def load_stimulus_features(root_data_dir, modality):
    """
    Load the stimulus features.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        Used feature modality.

    Returns
    -------
    features : dict
        Dictionary containing the stimulus features.

    """

    features = {}

    ### Load the visual features ###
    if modality == 'visual' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors', 'stimuli', 'stimulus_features', 'stimulus_features', 'pca',
            'friends_movie10', 'visual', 'features_train.npy')
        features['visual'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the audio features ###
    if modality == 'audio' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors', 'stimuli', 'stimulus_features', 'stimulus_features', 'pca',
            'friends_movie10', 'audio', 'features_train.npy')
        features['audio'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the language features ###
    if modality == 'language' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors', 'stimuli', 'stimulus_features', 'stimulus_features', 'pca',
            'friends_movie10', 'language', 'features_train.npy')
        features['language'] = np.load(stimuli_dir, allow_pickle=True).item()

    return features

def load_fmri(root_data_dir, subject):
    """
    Load the fMRI responses for the selected subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    subject : int
        Subject used to train and validate the encoding model.

    Returns
    -------
    fmri : dict
        Dictionary containing the  fMRI responses.

    """

    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    ### Output ###
    return fmri

def get_slices_context(U, Y, max_val, Y_subarray_length, context_steps, shift = 1, stacked = False):
    """Create sliding-window slices of inputs U and targets Y with optional context padding.
    """
    # repetition time 1.49 in fmri
    Y_max_start_index = max_val - Y_subarray_length + 1

    if stacked is True: 
        # context window timesteps are stacked, therefore Y during context window is not needed
        Y_subarrays = [Y[:, i:i+Y_subarray_length] for i in range(context_steps, Y_max_start_index, shift)]
    else: 
        # context window is needed in Y as whole segment is predicted to initialise carry correctly
        Y_subarrays = [Y[:, i:i+context_steps + Y_subarray_length] for i in range(0, Y_max_start_index- context_steps, shift)]
    
    U_subarrays = [U[:, i:i+context_steps + Y_subarray_length] for i in range(0, Y_max_start_index- context_steps, shift)]
    
    return np.stack(U_subarrays, axis=2), np.stack(Y_subarrays, axis=2)

def get_slices_per_session_context(Us, Ys, max_vals, t_length, context_length, shift = 1, stacked = False):
    """Slice multiple sessions/movies into context+prediction windows.
    """
    U_slices, Y_slices = [], []
    for i in range(len(Ys)):
        
        U_data, Y_data = get_slices_context(Us[i], Ys[i], max_vals[i], t_length, context_length, shift, stacked)

        Y_slices.append(Y_data.T)
        U_slices.append(U_data.T)
    return U_slices, Y_slices

def adjust_misalignment(X, Y):
    """Trim X or Y along time so both have the same number of timesteps."""
    if (np.shape(X)[1] < np.shape(Y)[1]):  
        return X, Y[:, :np.shape(X)[1]]
    return X[:, :np.shape(Y)[1]], Y

def get_timesteps_from_sec(length_in_s):
    TR = 1.49
    return math.ceil(length_in_s / TR) + 1 


def pad_arrays_to_max_T(arrays, array_len = None, shift = None):
    """Pad a list of (T, D) arrays with zeros so all share the same T.

    If array_len/shift are provided, T is adjusted upward to ensure clean divisibility
    for subsequent sliding-window slicing (prevents cutting off last timepoints).
    """
    # Get max T
    if array_len is None:
        n_timesteps = []
        for arr in arrays:
            n_timesteps.append(arr.shape[0])#
    else:
        # Needed for slicing approaches, Ensures clean divisibility so that no timepoint will be cutoff later during slices
        n_timesteps = []
        for arr in arrays:
            num_slices = math.ceil((arr.shape[0] - array_len) / shift) + 1
            adjusted_len = (num_slices - 1) * shift + array_len
            
            n_timesteps.append(adjusted_len)

    max_T = max(n_timesteps)
    
    padded_arrays = []
    for arr in arrays:
        pad_width = max_T - arr.shape[0]
        # Pad along the dimension T
        padded = jnp.pad(arr, ((0, pad_width), (0, 0)))
        padded_arrays.append(padded)

    return jnp.array(padded_arrays),n_timesteps, max_T

def align_features_and_fmri_samples(features_orig, fmri_orig, hrf_delay, movies, context_length_steps, segment_length_steps):
    """Align feature vectors and fMRI samples across splits and add context padding.

    Ensures feature time dimension matches fMRI time dimension (by truncating or
    repeating the last feature), then prepends context padding to both features
    (context+HRF delay) and fMRI (context only).

    Args:
        features_orig: Dict of modality -> dict of split -> (T, D_mod) feature arrays.
        fmri_orig: Dict of split -> (T, V) fMRI arrays.
        hrf_delay: HRF delay in timesteps (prepended to features).
        movies: List of movie identifiers to include (used to select splits).
        context_length_steps: Context length in timesteps (prepended to features/fMRI).

    Returns:
        (aligned_features, aligned_fmri): Lists of aligned arrays per split.
    """
    features = copy.deepcopy(features_orig)
    fmri = copy.deepcopy(fmri_orig)
    aligned_features = []
    aligned_fmri = []
    ### Loop across movies ###
    for movie in movies:

        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == 'friends':
            id = movie[8:]
        elif movie[:7] == 'movie10':
            id = movie[8:]
        movie_splits = [key for key in fmri if id in key[:len(id)]]

        ### Loop over movie splits ###
        for split in movie_splits:

            ### Extract the fMRI ###
            fmri_split = fmri[split] 
                     
            f_all = []
            ### Loop across modalities ###
            for mod in features:
                misaligned_end_steps = jnp.shape(fmri_split.T)[1]  -jnp.shape(features[mod][split].T)[1]
               
                if misaligned_end_steps > 0:
                    f_end = [features[mod][split][-1, :]] * misaligned_end_steps
                    # Additions are appended as additional timesteps at end of feature vectors
                    features[mod][split] = np.concatenate([features[mod][split], f_end], axis=0)
                    
                else: 
                    # Reduce feature vector size to fmri size
                    features[mod][split] =  features[mod][split][:fmri_split.shape[0], :]
          
                # To predict first fmri samples, context vectors are created by padding with the first timestep 
                # when not enough timesteps are available for the desired context window
                if context_length_steps+hrf_delay > 0:
                    f = [features_orig[mod][split][0, :]] * (context_length_steps+hrf_delay)
            
                # Additions are appended as additional timesteps at front of feature vectors
                features[mod][split] = np.concatenate([f, features[mod][split]], axis=0)
                
                f_all.append(features[mod][split])
                
            # fmri vector is extended by context length steps in front 
            # (those steps will not be predicted but will (depending on model type be used as context))
            f = [fmri_split[0, :]] * context_length_steps
            if context_length_steps > 0:
                fmri_split = np.concatenate([f, fmri_split], axis=0)
            features_split = np.hstack((f_all))
    
            aligned_fmri.append(fmri_split)
            aligned_features.append(features_split)
            

    return aligned_features, aligned_fmri

# ensure ordering of dictionaries
def sort_key(k):
    """Key function to sort split IDs (friends seasons/episodes, then movie10 labels, then fallback)."""
    
    # Match 's01e01a' style
    match_tv = re.match(r's(\d+)e(\d+)([a-z])', k)
    if match_tv:
        season = int(match_tv.group(1))
        episode = int(match_tv.group(2))
        part = match_tv.group(3)
        return (0, season, episode, part)  
    

    # Match 'figures09' or 'life03' style
    match_label = re.match(r'([a-zA-Z]+)(\d+)', k)
    if match_label:
        label = match_label.group(1)
        number = int(match_label.group(2))
        return (1, label, number)

    # Fallback for anything else
    return (2, k)
