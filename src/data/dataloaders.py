"""
Module for loading and batching fMRI + stimulus data for training models on temporal tasks
(e.g., AFM, SFM, LatentSDE). Supports single- and multi-subject datasets, infinite/random
batches, and deterministic validation splits.
"""
import jax
import jax.numpy as jnp
from typing import Iterator
import jax_dataloader as jdl
import jax.random as jrandom
import numpy as np
import logging
import utils.utils_combi as utils
import os
from typing import List, Tuple, Optional, Iterator, Union

VAL_SPLIT = 0.1
####
# Dataloading utils
####
def get_split_list(movies: list[str], fmri: dict[str, jnp.ndarray]) -> list[str]:
    """
    Get all data keys in the fmri dict that match provided movies.
    """
    split_list = []
    for movie in movies:

        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == 'friends':
            id = movie[8:]
        elif movie[:7] == 'movie10':
            id = movie[8:]
        for key in fmri:
            if id in key[:len(id)]:
                split_list.append(key) 
    return split_list

def get_train_val_split_indices(runconfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly split split_list into training and validation indices.
    """
    fmri = utils.load_fmri(runconfig.root_dir, 1) # always based on subj 1 bc no missing data here
    split_list = get_split_list(runconfig.movies_train, fmri)
    num_samples = len(split_list)
    key = jrandom.PRNGKey(390)
    random_indices = jrandom.permutation(key, jnp.arange(num_samples))
    num_val_samples = int(num_samples * VAL_SPLIT)
    val_indices = random_indices[:num_val_samples]  
    train_indices = random_indices[num_val_samples:] 
    return np.sort(train_indices), np.sort(val_indices), {idx:item for idx, item in enumerate(split_list)}

def apply_voxel_selection(fmri: dict[str, jnp.ndarray], area: Optional[str]) -> dict[str, jnp.ndarray]:
    """
    Applies voxel selection based on the specified brain area.
    Currently supports:
        - "V1": Primary visual cortex
    """
    if area == "V1":
        selected_voxels = np.concatenate((np.arange(1, 67), np.arange(501, 569)))
        for (key, value) in fmri.items():
            fmri[key] = value[:, selected_voxels]
    return fmri

def sort_data(fmri: dict[str, jnp.ndarray], features: dict[str, dict]) -> Tuple[dict[str, jnp.ndarray], dict[str, dict]]:
    """
    Ensure ordering of samples
    """
    for (key, value) in features.items():
        sorted_data = dict(sorted(value.items(), key=lambda item: utils.sort_key(item[0])))
        features[key] = sorted_data
    
    fmri = dict(sorted(fmri.items(), key=lambda item: utils.sort_key(item[0])))

    return fmri, features

####
# Data loading
####
def prepare_test_sampling_s7(
    subject_id: int,
    root_dir: str,
    hrf_delay: int,
    context_length_steps: int
) -> Tuple[jnp.ndarray, None, dict[str, int]]:
    """
    Prepares test data for Season 7 ("friends-s07"), where only stimulus features are available
    and no fMRI recordings are provided.
    """
    features_raw = utils.load_stimulus_features_friends_s7(root_dir)
    samples_path = os.path.join(root_dir, 'algonauts_2025.competitors',
            'fmri', f'sub-0{subject_id}', 'target_sample_number',
            f'sub-0{subject_id}_friends-s7_fmri_samples.npy')
    frame_counts = np.load(samples_path, allow_pickle=True).item()  

    features_aligned = utils.align_features_and_fmri_samples_friends_s7(features_raw, frame_counts, 
                                                                        hrf_delay, context_length_steps)
    controls, _, _ = utils.pad_arrays_to_max_T(features_aligned)

    return controls, None, frame_counts

def get_data(
    features: dict,
    fmri: dict,
    runconfig,
    movies: List[str],
    train: bool = True,
    selectors: Optional[List[str]] = None,
    val_indices: Optional[np.ndarray] = None
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray, np.ndarray],  # train=True
        Tuple[jnp.ndarray, jnp.ndarray, dict[str, int]]  # train=False
    ]:
    """
    Loads, aligns, and pads stimulus and fMRI data for training or sampling.

    Behavior depends on whether the function is being used for training or sampling:
    - For training (train=True): Returns padded [N, T, D] arrays for model input.
    - For sampling (train=False): Slices each sequence into fixed-length segments for prediction.

    Args:
        features (dict): Stimulus feature dictionary.
        fmri (dict): fMRI data dictionary.
        runconfig: Configuration object with model and preprocessing parameters.
        movies (list[str]): List of movie/episode names to load.
        train (bool): Whether to load training data (True) or prepare sampling data (False).
        selectors (list[str] | None): Optional list of episode keys used for frame count extraction.
        val_indices (np.ndarray | None): Optional subset of indices (used for val/test splits).

    Returns:
        If train == True:
            Tuple:
                - controls (jnp.ndarray): Stimulus features (padded)
                - fmri (jnp.ndarray): fMRI signals (padded)
                - train_steps (np.ndarray): Number of valid steps per sample

        If train == False:
            Tuple:
                - controls (jnp.ndarray): Sliced stimulus sequences (sliced)
                - fmri (jnp.ndarray): Sliced fMRI sequences (sliced)
                - frame_counts (dict): Frame count metadata per episode (used for evaluation)
    """
    aligned_features, aligned_fmri = utils.align_features_and_fmri_samples(features, fmri, runconfig.hrf_delay, movies, runconfig.context_length_steps, runconfig.segment_length_steps)
    
    is_seq_model = runconfig.framework in ("latentsde", "sfm") or not train
    array_len = runconfig.context_length_steps + runconfig.segment_length_steps if is_seq_model else None
    shift = runconfig.segment_length_steps if is_seq_model else None

    controls, train_steps, _ = utils.pad_arrays_to_max_T(aligned_features, array_len, shift )
    fmri_pad, max_vals, _ = utils.pad_arrays_to_max_T(aligned_fmri,  array_len, shift)
    
    # preparation for sampling
    if train is False: 
        controls = jnp.swapaxes(controls, 1, 2)
        fmri_pad = jnp.swapaxes(fmri_pad, 1, 2)

        shift = runconfig.segment_length_steps
        if val_indices is not None:
            controls = np.array(controls)[val_indices]
            fmri_pad = np.array(fmri_pad)[val_indices]
            max_vals= np.array(max_vals)[val_indices]
        controls_sliced, fmri_sliced = utils.get_slices_per_session_context(controls, fmri_pad,max_vals, runconfig.segment_length_steps, runconfig.context_length_steps, shift=runconfig.segment_length_steps)
        sampling_fmri_frame_counts = {
            str(episode): jnp.shape(fmri[episode])[0]
            for episode in selectors
        }
        return  controls_sliced, fmri_sliced, sampling_fmri_frame_counts

    return controls, fmri_pad, np.array(train_steps)

def load_subject_data(runconfig,
    modality: str = "all",
    area: Optional[str] = None,
    train_indices = None,
    val_indices = None, 
    epi_mapping = None,
    load_multiple = False,
    ) -> dict:
    """
    Loads and prepares training, validation, and sampling data for a single subject.

    Args:
        runconfig: Configuration object with attributes such as:
            - subject_id: Subject identifier
            - hrf_delay: Delay in fMRI response due to hemodynamic lag
            - context_length_steps: Number of time steps for context input
            - segment_length_steps: Length of prediction segment
            - framework: Model type ("sfm", "afm ", "latentsde")
            - hyperopt: If True, use validation for testing during hyperparameter search
        modality (str): Feature modality to use (e.g., "visual", "audio", or "all")
        area (str or None): Region of interest in the brain (e.g., "V1"), filters voxel selection

    Returns:
        dict: Dictionary containing training, validation, and optionally test/sampling datasets:
            {
                "train": {
                    "fmri": fMRI signals,
                    "control": stimulus signals,
                    "n_timesteps": valid time steps per sample,
                    "subject_ids": subject ID array
                },
                "val": { ... same as train ... },
                "train_sampling": {
                    "fmri": full fMRI dictionary,
                    "fmri_adj": adjusted/segmented fMRI tensor,
                    "control": adjusted stimulus array,
                    "fmri_frame_counts": dict of frame lengths per episode
                },
                "test_sampling": {
                    "fmri": full fMRI dictionary (if available),
                    "fmri_adj": test fMRI segments (or None if not available),
                    "control": adjusted stimulus array,
                    "fmri_frame_counts": dict of frame lengths per episode
                }
            }
    """
    # Load and sort
    features = utils.load_stimulus_features(runconfig.root_dir,modality)
    fmri = utils.load_fmri(runconfig.root_dir, runconfig.subject_id)
    fmri =  apply_voxel_selection(fmri, area)
    fmri, features = sort_data(fmri, features)

    # Load training data
    train_controls, train_fmri, train_steps = get_data(features, fmri, runconfig, runconfig.movies_train)
    
    # Use indices to create train/val splits
    split_list = get_split_list(runconfig.movies_train, fmri)
    if train_indices is None or val_indices is None: # to ensure the same validation dataset across subjects despite missing data fo subj 3 and 5
        train_indices, val_indices, epi_mapping = get_train_val_split_indices(runconfig)

      
    # Due to missing episodes for subj 3 and 5, remapping of training and val indices is necessary to ensure the same validation dataset usage
    epi_mapping_subj = {item:idx for idx, item in enumerate(split_list)}

    adj_ind = []
    for i in train_indices:
        if epi_mapping[i] in epi_mapping_subj: 
            adj_ind.append(epi_mapping_subj[epi_mapping[i]])
    train_indices_subj = np.array(adj_ind)
    adj_ind = []
    for i in val_indices:
        adj_ind.append(epi_mapping_subj[epi_mapping[i]])
    val_indices_subj = np.array(adj_ind)
    train_fmri, val_fmri = train_fmri[train_indices_subj], train_fmri[val_indices_subj]
    train_controls, val_controls = train_controls[train_indices_subj], train_controls[val_indices_subj]
    train_steps, val_steps = train_steps[train_indices_subj], train_steps[val_indices_subj]
       
    # Load sampling data
    train_sample_controls, train_sample_fmri, train_sampling_fmri_frame_counts = get_data(features, fmri, runconfig,[runconfig.movies_train[0]], False, get_split_list([runconfig.movies_train[0]], fmri))
    if runconfig.hyperopt:
        test_controls, test_fmri, test_fmri_frame_counts = get_data(features, fmri, runconfig, runconfig.movies_train, False, np.array(split_list)[val_indices_subj], val_indices_subj)

    elif runconfig.movies_test[0] != "friends-s07":
        test_controls, test_fmri, test_fmri_frame_counts = get_data(features, fmri, runconfig, runconfig.movies_test, False,  get_split_list(runconfig.movies_test, fmri))
    else:
        test_controls, test_fmri, test_fmri_frame_counts = prepare_test_sampling_s7(runconfig.subject_id, runconfig.root_dir, runconfig.hrf_delay, runconfig.context_length_steps)

    data = {
            "train": {
                "fmri": train_fmri,
                "control": train_controls,
                "n_timesteps": train_steps,
                "subject_ids": jnp.full((train_fmri.shape[0],), runconfig.subject_id)
            },
            "val": {
                "fmri": val_fmri,
                "control": val_controls,
                "n_timesteps": val_steps, 
                "subject_ids": jnp.full((val_fmri.shape[0],), runconfig.subject_id)
            },
            "train_sampling": {
                "fmri": fmri,
                "fmri_adj": train_sample_fmri,
                "control": train_sample_controls,
                "fmri_frame_counts": train_sampling_fmri_frame_counts
            },
            "test_sampling": {
                "fmri": fmri,
                "fmri_adj": test_fmri,
                "control": test_controls,
                "fmri_frame_counts": test_fmri_frame_counts
            }
        }
    
    if load_multiple:
        return data, train_indices, val_indices, epi_mapping
    else:
        return data


def load_multiple_subjects_data(runconfig,
        subjects: List[int],
        modality: str = "all",
        area: Optional[str] = None
    ) -> dict:
    """
    Loads training/validation data for multiple subjects and concatenates them.
    """
    all_train_fmri, all_train_control, all_n_timesteps, all_subject_ids = [], [], [], []
    all_val_fmri, all_val_control, all_n_timesteps_val, all_subject_ids_val = [], [], [], []
    train_indices, val_indices, epi_mapping = None, None, None
    for subject in subjects:
        runconfig.subject_id = subject
        data, train_indices, val_indices, epi_mapping = load_subject_data(runconfig, modality, area, train_indices, val_indices, epi_mapping, load_multiple = True)

        n_train = data["train"]["fmri"].shape[0]
        n_val = data["val"]["fmri"].shape[0]

        # Append training data
        all_train_fmri.append(data["train"]["fmri"])
        all_train_control.append(data["train"]["control"])
        all_n_timesteps.append(data["train"]["n_timesteps"])
        all_subject_ids.append(data["train"]["subject_ids"])

        # Append validation data
        all_val_fmri.append(data["val"]["fmri"])
        all_val_control.append(data["val"]["control"])
        all_n_timesteps_val.append(data["val"]["n_timesteps"])
        all_subject_ids_val.append(data["train"]["subject_ids"])

    # Stack across all subjects
    combined_data = {
        "train": {
            "fmri": jnp.concatenate(all_train_fmri, axis=0),
            "control": jnp.concatenate(all_train_control, axis=0),
            "n_timesteps": jnp.concatenate(all_n_timesteps, axis=0),
            "subject_ids": jnp.concatenate(all_subject_ids, axis=0),
        },
        "val": {
            "fmri": jnp.concatenate(all_val_fmri, axis=0),
            "control": jnp.concatenate(all_val_control, axis=0),
            "n_timesteps": jnp.concatenate(all_n_timesteps_val, axis=0),
            "subject_ids": jnp.concatenate(all_subject_ids_val, axis=0),
        }
    }

    return combined_data

####
# Dataloaders
####

def build_batch(
    trajectories: jnp.ndarray,
    control: jnp.ndarray,
    subject_ids: Union[jnp.ndarray, np.ndarray, List[int]],
    traj_indices: jnp.ndarray,
    start_indices: jnp.ndarray,
    context_length: int,
    framework: str,
    prediction_horizon: int = 1
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],  # afm
        Tuple[np.ndarray, np.ndarray],  # sfm
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]   # latentsde
    ]:
    """
    Builds a single training batch given trajectory and control signals.

    Args:
        trajectories: [N, T, D] array of fMRI signals
        control: [N, T, C] array of stimulus/control inputs
        subject_ids: list or array of subject identifiers
        traj_indices: indices of the trajectories in the batch
        start_indices: start indices for each trajectory slice
        context_length: number of past time steps for context
        framework: model framework ("afm", "sfm", "latentsde")
        prediction_horizon: number of future steps to predict

    Returns:
        Data batch formatted according to model framework
    """
    state_dim = trajectories.shape[-1]
    control_dim = control.shape[-1]
    next_indices = start_indices + context_length

    def get_single_context(traj_idx, start_idx):
        traj_slice = jax.lax.dynamic_slice(
            trajectories,
            (traj_idx, start_idx, 0),
            (1, context_length, state_dim)
        )
        control_slice = jax.lax.dynamic_slice(
            control,
            (traj_idx, start_idx, 0),
            (1, context_length, control_dim)
        )
        return jnp.squeeze(traj_slice, axis=0), jnp.squeeze(control_slice, axis=0)

    def get_single_next(traj_idx, next_idx):
        x_slice = jax.lax.dynamic_slice(
            trajectories,
            (traj_idx, next_idx, 0),
            (1, prediction_horizon, state_dim)
        )
        c_slice = jax.lax.dynamic_slice(
            control,
            (traj_idx, next_idx, 0),
            (1, prediction_horizon, control_dim)
        )
        x_slice = jnp.squeeze(x_slice, axis=0)
        c_slice = jnp.squeeze(c_slice, axis=0)
        if prediction_horizon == 1 and framework == "afm":
            x_slice = jnp.squeeze(x_slice, axis=0)
            c_slice = jnp.squeeze(c_slice, axis=0)
        return x_slice, c_slice

    get_context_fn = jax.vmap(get_single_context)
    get_next_fn = jax.vmap(get_single_next)

    x_prev, c_prev = get_context_fn(traj_indices, start_indices)
    x_next, c_next = get_next_fn(traj_indices, next_indices)
    subject_ids = jnp.array(subject_ids[traj_indices])

    if framework == "afm":
        return x_prev, c_prev, x_next, subject_ids  # 2D x_next
    elif framework == "sfm":
        return x_prev, np.concatenate([c_prev, c_next], axis = 1), x_next
       # return np.concatenate([x_prev, x_next], axis = 1), np.concatenate([c_prev, c_next], axis = 1)
    elif framework == "latentsde":
        return x_prev, c_prev, x_next, c_next, subject_ids

   
def create_infinite_dataloader(
    trajectories: jnp.ndarray,
    control: jnp.ndarray,
    n_timesteps: jnp.ndarray,
    subject_ids: Union[jnp.ndarray, np.ndarray],
    context_length: int,
    framework: str,
    batch_size: int = 32,
    prediction_horizon: int = 1,
    seed: int = 0
) -> Iterator:
    """
    Creates an infinite generator of random training batches.

    Args:
        trajectories: [N, T, D] array of fMRI signals
        control: [N, T, C] array of stimulus/control inputs
        subject_ids: list or array of subject identifiers
        traj_indices: indices of the trajectories in the batch
        start_indices: start indices for each trajectory slice
        context_length: number of past time steps for context
        framework: model framework ("afm", "sfm", "latentsde")
        prediction_horizon: number of future steps to predict
        seed: random seed

    Returns:
        Infinite iterator of training batches.
    """
    n_samples = trajectories.shape[0]
    rng = jax.random.PRNGKey(seed)
    
    while True:
        rng, rng_traj, rng_start = jax.random.split(rng, 3)

        traj_indices = jax.random.randint(
            rng_traj, (batch_size,), 0, n_samples
        )
        max_starts = n_timesteps[traj_indices] - context_length - prediction_horizon + 1
        start_indices = jax.random.randint(
            rng_start, (batch_size,), 0, max_starts
        )

        yield build_batch(
            trajectories, control,subject_ids,
            traj_indices, start_indices,
            context_length,
            framework,
            prediction_horizon
        )


class FiniteDataLoader:
    """
    A dataloader that yields finite batches for validation or evaluation.
    """
    def __init__(
        self,
        trajectories: jnp.ndarray,
        control: jnp.ndarray,
        n_timesteps: jnp.ndarray,
        subject_ids: Union[jnp.ndarray, np.ndarray],
        context_length: int,
        batch_size: int,
        framework: str,
        prediction_horizon: int = 1
    ) -> None:
        self.trajectories = trajectories
        self.control = control
        self.n_timesteps = n_timesteps
        self.subject_ids = subject_ids
        self.context_length = context_length
        self.batch_size = batch_size
        self.prediction_horizon = prediction_horizon
        self.framework = framework
        self.subject_ids

    def create_finite_dataloader(self) -> Iterator:
        n_samples = self.trajectories.shape[0]

        valid_indices = []
        for i in range(n_samples):
            max_start = self.n_timesteps[i] - self.context_length - self.prediction_horizon + 1
            for t in range(max_start):
                valid_indices.append((i, t))

        total = len(valid_indices)
        for i in range(0, total, self.batch_size):
            batch = valid_indices[i:i + self.batch_size]
            traj_indices = jnp.array([x[0] for x in batch])
            start_indices = jnp.array([x[1] for x in batch])

            yield build_batch(
                self.trajectories, self.control, self.subject_ids,
                traj_indices, start_indices,
                self.context_length,
                self.framework,
                self.prediction_horizon
            )
           
        
    def __iter__(self) -> Iterator:
        return self.create_finite_dataloader()


def create_dataloaders( data: dict,
        runconfig,
        prediction_horizon: int,
        framework: str,
        seed: int = 42
    ) -> Union[
        Tuple[Iterator, FiniteDataLoader],
        Tuple[Iterator, FiniteDataLoader, List[jdl.DataLoader], List[jdl.DataLoader]]
    ]:
    """
    Creates dataloaders for training and validation sets.

    Returns:
        train_loader, val_loader, [optional sampling loaders]
    """
    train_traj = data["train"]["fmri"]
    train_control = data["train"]["control"]
    n_timesteps = jnp.array(data["train"]["n_timesteps"])
    val_traj = data["val"]["fmri"]
    val_control = data["val"]["control"]
    n_timesteps_val = jnp.array(data["val"]["n_timesteps"])

    train_subject_ids = data["train"]["subject_ids"]
    val_subject_ids = data["val"]["subject_ids"]

    #Custom infinte dataloader for training
    train_loader = create_infinite_dataloader(trajectories=train_traj, 
                                        control=train_control,    
                                        n_timesteps=n_timesteps,  
                                        subject_ids = train_subject_ids,  
                                        context_length=runconfig.context_length_steps,
                                        batch_size=runconfig.batch_size,
                                        framework = framework,
                                        prediction_horizon=prediction_horizon,
                                        seed=seed)
    
    val_loader = FiniteDataLoader(trajectories=val_traj, 
                                        control=val_control,    
                                        n_timesteps=n_timesteps_val,   
                                        subject_ids = val_subject_ids, 
                                        context_length=runconfig.context_length_steps,
                                        prediction_horizon=prediction_horizon,
                                        framework = framework,
                                        batch_size=runconfig.batch_size)

    runconfig.dim_y = np.shape(train_traj)[2]
    runconfig.dim_u = np.shape(train_control)[2]
        
    # Sampling Dataloaders
    if "train_sampling" in data:
        train_control2 = data["train_sampling"]["control"]
        train_traj2 = data["train_sampling"]["fmri_adj"]
        test_control = data["test_sampling"]["control"]
        test_traj = data["test_sampling"]["fmri_adj"]

        def make_dataloader(traj, control):
    
            prev_traj = traj[:, :runconfig.context_length_steps, :]
            next_traj = traj[:, runconfig.context_length_steps:, :]
            
            if runconfig.framework == "latentsde":

                prev_control = control[:, :runconfig.context_length_steps, :]
                next_control = control[:, runconfig.context_length_steps:, :]
                return jdl.DataLoader(jdl.ArrayDataset(prev_traj, prev_control, next_traj, next_control), batch_size=32, shuffle=False, backend='jax')
            elif runconfig.framework == "sfm":
                return jdl.DataLoader(jdl.ArrayDataset(prev_traj, control, next_traj), batch_size=32, shuffle=False, backend='jax')

            else: 
                return jdl.DataLoader(jdl.ArrayDataset(prev_traj, control, next_traj), batch_size=32, shuffle=False, backend='jax')
        

        test_loader_per_epi = []
        
        for i in range(len(test_traj)):
            test_loader_per_epi.append(make_dataloader(test_traj[i], test_control[i])) 
        logging.info(f"Number of test loaders {len(test_loader_per_epi)}")
        train_loader2_per_epi = []
        for i in range(len(train_traj2)):
            train_loader2_per_epi.append(make_dataloader(train_traj2[i], train_control2[i])) 

        return  train_loader, val_loader, test_loader_per_epi, train_loader2_per_epi
    else: return train_loader, val_loader