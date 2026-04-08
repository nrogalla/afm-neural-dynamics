import numpy as np
import logging
from data.dataloaders import load_subject_data
import utils.runconfigs as rc
import jax.numpy as jnp
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
import jax


def _roi_counts(area_ids, n_areas=None):
    """
    Returns counts per area
    """
    area_ids = jnp.asarray(area_ids).astype(jnp.int32)
    if n_areas is None:
        n_areas = int(jnp.max(area_ids)) + 1
    counts = jnp.zeros((n_areas,), dtype=jnp.float32).at[area_ids].add(1.0)
    return counts

def reduce_parcel_vector_to_areas(vecP, area_ids, n_areas=None, reduce="mean"):
    """
    Reduce parcel/voxel-level values into area-level values.
    """
    area_ids = jnp.asarray(area_ids).astype(jnp.int32)
    vecP = jnp.asarray(vecP)
    if n_areas is None:
        n_areas = int(jnp.max(area_ids)) + 1

    sums = jnp.zeros((n_areas,), dtype=vecP.dtype).at[area_ids].add(vecP)
    if reduce == "sum":
        return sums
    counts = _roi_counts(area_ids, n_areas)
    return sums / jnp.maximum(counts, 1.0)

def reduce_parcel_matrix_to_areas(matNP, area_ids, n_areas=None, reduce="mean"):
    """
    Reduce a matrix of parcel/voxel-level vectors into area-level vectors row-wise.
    """
    area_ids = jnp.asarray(area_ids).astype(jnp.int32)
    matNP = jnp.asarray(matNP)
    if n_areas is None:
        n_areas = int(jnp.max(area_ids)) + 1

    def _one_row(rowP):
        return reduce_parcel_vector_to_areas(rowP, area_ids, n_areas, reduce=reduce)
    return jax.vmap(_one_row)(matNP)


def jax_perm_diff_by_area_masked(Y, A, B, M, area_ids, block_len: int, n_perm: int, seed: int = 0):
    """
    Compute a voxel/parcel-level permutation difference map (A vs B), then aggregate
    to brain areas and compute two-sided p-values per area.

    Parameters
    ----------
    Y : jax.Array, shape (S, T, P)
        Ground-truth fMRI data across subjects (S), time (T), and parcels/voxels (P).

    A, B : jax.Array, shape (S, T, P)
        Two competing model predictions aligned with Y.

    M : jax.Array, shape (S, T)
        Boolean mask indicating valid timesteps (True = valid). Used to handle
        variable-length episode concatenations or padding.

    area_ids : array-like, shape (P,)
        Integer mapping from each parcel/voxel to an area id.

    block_len : int
        Block length (in timesteps) used to create swap masks for the blocked
        permutation scheme.

    n_perm : int
        Number of permutations to generate for the null distribution.

    seed : int, default=0
        PRNG seed for permutations.

    Returns
    -------
    obs_area : jax.Array, shape (n_areas,)
        Observed area-level difference map.

    p_area : jax.Array, shape (n_areas,)
        Two-sided p-values per area computed from the null distribution.

    null_area : jax.Array, shape (n_perm, n_areas)
        Area-level null distribution.
    """
    obs_map, p_map_dummy, null_map = jax_perm_diff_map_masked(
        Y, A, B, M, block_len=block_len, n_perm=n_perm, seed=seed
    )  
    area_ids_j = jnp.asarray(area_ids)
    obs_area  = reduce_parcel_vector_to_areas(obs_map, area_ids_j)           
    null_area = reduce_parcel_matrix_to_areas(null_map, area_ids_j)         

    # two-sided p per area
    abs_obs  = jnp.abs(obs_area)[None, :]         
    abs_null = jnp.abs(null_area)                 
    p_area = (jnp.sum(abs_null >= abs_obs, axis=0) + 1.0) / (n_perm + 1.0)   

    return obs_area, p_area, null_area


def perf_vs_chance_area_from_maps(obs_map, null_map, area_ids, two_sided=False):
    """
    Aggregate parcel-level observed and null maps to areas, then compute p-values.

    Parameters
    ----------
    obs_map : jax.Array, shape (P,)
        Observed parcel/voxel-level statistic map (e.g., mean correlation per voxel).

    null_map : jax.Array, shape (N, P)
        Null distribution maps (N permutations) at parcel/voxel level.

    area_ids : array-like, shape (P,)
        Integer mapping from each parcel/voxel to an area id.

    two_sided : bool, default=False
        If True, compute two-sided p-values using absolute values.
        If False, compute one-sided p-values testing null >= observed.

    Returns
    -------
    obs_area : jax.Array, shape (n_areas,)
        Observed area-level statistic.

    p_area : jax.Array, shape (n_areas,)
        Area-level p-values.

    null_area : jax.Array, shape (N, n_areas)
        Area-level null distribution.
    """
    area_ids_j = jnp.asarray(area_ids)

    obs_area  = reduce_parcel_vector_to_areas(obs_map, area_ids_j)    
    null_area = reduce_parcel_matrix_to_areas(null_map, area_ids_j) 

    if two_sided:
        thresh = jnp.abs(obs_area)[None, :]
        p_area = (jnp.sum(jnp.abs(null_area) >= thresh, axis=0) + 1.0) / (null_area.shape[0] + 1.0)
    else:
        thresh = obs_area[None, :]
        p_area = (jnp.sum(null_area >= thresh, axis=0) + 1.0) / (null_area.shape[0] + 1.0)

    return obs_area, p_area, null_area

def pearsonr_cols(X, Y, mask, eps=1e-12):
    """
    Compute masked Pearson correlation between corresponding columns of X and Y.

    This computes Pearson r for each feature/voxel (column) across time, using a
    boolean mask to ignore invalid timesteps.

    Parameters
    ----------
    X, Y : array-like, shape (T, P)
        Two time-by-voxel matrices. Correlation is computed per voxel (per column).

    mask : array-like, shape (T,)
        Boolean mask over time. True indicates valid timesteps.

    eps : float, default=1e-12
        Small constant to avoid division by zero.

    Returns
    -------
    r : jax.Array, shape (P,)
        Pearson correlation coefficient per column/voxel.
    """
    Mw = mask[..., None]
    X = jnp.asarray(X)
    cnt = jnp.sum(Mw, axis=0, keepdims=True) 
    cnt = jnp.maximum(cnt, 1.0)
   
    X_mean = jnp.sum(X * Mw, axis=0, keepdims=True) / cnt
    Y_mean = jnp.sum(Y * Mw, axis=0, keepdims=True) / cnt

    X0 = (X - X_mean) * Mw
    Y0 = (Y - Y_mean) * Mw
    num = jnp.sum(X0 * Y0, axis=0)
    denom = jnp.sqrt(jnp.sum(X0 * X0, axis=0) * jnp.sum(Y0 * Y0, axis=0))
   
    return num / (denom + eps)


def build_block_swap_masks(key, n_perm, T, block_len):
    """
    Build boolean swap masks over time for blocked A/B swapping.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key.

    n_perm : int
        Number of permutations.

    T : int
        Number of timesteps.

    block_len : int
        Block length in timesteps.

    Returns
    -------
    masks : jax.Array, shape (n_perm, T)
        Boolean masks; True indicates "swap A and B" at that timestep.
    """
    n_blocks = (T + block_len - 1) // block_len
    key_blocks, key_pad = jax.random.split(key)
    
    keys = jax.random.split(key_blocks, n_perm)
   
    block_flags = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5, (n_blocks,)))(keys)  # (n_perm, n_blocks)
   
    masks = jax.vmap(lambda row: jnp.repeat(row, block_len)[:T])(block_flags)          # (n_perm, T)
    return masks

def build_circular_shifts(key, n_perm, T):
    """
    Sample random circular time shifts for permutation testing.
    """
    shifts = jax.random.randint(key, shape=(n_perm,), minval=0, maxval=T)
    return shifts

def roll_time(x, shift):
    """
    Circularly shift time axis
    """
    return jnp.roll(x, shift=shift, axis=1)

def compute_pearson_score_full(fmri_test, fmri_test_pred, mask):
    """
    Compute mean masked Pearson correlation across voxels/parcels for one subject.
    """
    r_per_voxel = pearsonr_cols(fmri_test, fmri_test_pred, mask)  
    return jnp.round(jnp.mean(r_per_voxel), 3)              

@jax.jit
def compute_pearson_score_per_subject_masked(Y, Y_pred, M):
    """
    Compute Pearson correlation scores per subject (and their mean), with masking.
    """
    r_subj_voxel = jax.vmap(compute_pearson_score_full, in_axes=(0, 0, 0))(Y, Y_pred, M)  # (S)

    return jnp.mean(r_subj_voxel), r_subj_voxel


def jax_perm_diff_of_means_masked(Y, A, B, M, block_len: int, n_perm: int, seed: int = 0, subject_level = False):
    """
    Permutation test on the difference in mean performance between two models.

    Observed statistic:
      score(Y, A) - score(Y, B)

    Null distribution:
      - circularly shift Y/A/B/M by a random amount (preserves temporal autocorrelation)
      - swap A and B within random time blocks using a Bernoulli block mask
      - recompute statistic

    Parameters
    ----------
    Y : jax.Array, shape (S, T, P)
        Ground truth.

    A, B : jax.Array, shape (S, T, P)
        Model predictions.

    M : jax.Array, shape (S, T)
        Boolean mask.

    block_len : int
        Block length for swap mask.

    n_perm : int
        Number of permutations.

    seed : int, default=0
        PRNG seed.

    subject_level : bool, default=False
        If False, statistic is the group mean score difference (scalar).
        If True, statistic is per-subject score difference (shape (S,)).

    Returns
    -------
    obs : jax.Array
        Observed statistic (scalar or (S,)).

    p : jax.Array
        Two-sided p-value(s), matching `obs` shape.

    null : jax.Array, shape (n_perm, ...)  (or (n_perm, S) in subject_level mode)
        Null distribution of the statistic.
    """
    S, T, P = Y.shape

    obs = compute_pearson_score_per_subject_masked(Y, A, M)[int(subject_level)] - \
          compute_pearson_score_per_subject_masked(Y, B, M)[int(subject_level)]

    key = jax.random.PRNGKey(seed)
    key_shifts, key_masks = jax.random.split(key)
    shifts = build_circular_shifts(key_shifts, n_perm, T)             
    masks  = build_block_swap_masks(key_masks, n_perm, T, block_len)    

    def one_perm(shift, swap_mask_time):
        Ys = roll_time(Y, shift)
        As = roll_time(A, shift)
        Bs = roll_time(B, shift)
        Ms = roll_time(M, shift)

        sm = swap_mask_time[None, :, None]  

        # Swap A/B where sm is True
        Ap = jnp.where(sm, Bs, As)
        Bp = jnp.where(sm, As, Bs)

        return compute_pearson_score_per_subject_masked(Ys, Ap, Ms)[int(subject_level)] - \
               compute_pearson_score_per_subject_masked(Ys, Bp, Ms)[int(subject_level)]

    null = []
    for i, (shift, sm) in enumerate(zip(shifts, masks)):
        null.append(one_perm(int(shift), sm))
        if i % 1000 == 0:
            logging.info(i)
    null = jnp.array(null)

    abs_obs = jnp.abs(obs)
    p = (jnp.sum(jnp.abs(null) >= abs_obs, axis = 0) + 1.0) / (n_perm + 1.0)
    return obs, p, null


def save_null_plot(null, obs, p, out_path="perm_null_plot.png", bins=40):
    """
    Save a histogram plot of a permutation null distribution with the observed value.
    """
    null = np.asarray(null).ravel()
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))

   
    ax.hist(null, bins=bins, alpha=0.8, edgecolor="black", label="Null distribution")
    ax.axvline(obs, color="red", linestyle="--", linewidth=2, label=f"Observed difference = {obs:.4g}")

    ax.set_xlabel("Correlation difference")
    ax.set_ylabel("Count")
    ax.set_title(f"Null distribution (n={len(null)}) — p = {p:.3g}")

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
def pad_and_make_mask(list_of_arrays):
    """
    Pad a list of variable-length time series to a common length and build a mask.
    """
    S = len(list_of_arrays)
    P = list_of_arrays[0].shape[1]
    T_max = max(a.shape[0] for a in list_of_arrays)

    X_pad = np.zeros((S, T_max, P), dtype=np.float32)
    M    = np.zeros((S, T_max),    dtype=bool)
    for i, a in enumerate(list_of_arrays):
        t = a.shape[0]
        X_pad[i, :t, :] = a
        M[i, :t] = True
    return jnp.asarray(X_pad), jnp.asarray(M)

def get_preds_and_y(root_dir, subjects, framework1, framework2, modeltype1, modeltype2, dataset, hyper1, hyper2):
    """
    Load ground-truth fMRI and two sets of model predictions for multiple subjects.

    This function loads:
      - subject data via `load_subject_data`
      - prediction dicts saved as `preds_test_sampling.npy` for two model configurations
    It then concatenates across episodes and trims 5 frames at start and end.
    """
    runconfig = rc.BaseRunConfigs(None, None)
    # runconfig.root_dir = "C:/Users/rogal/Documents/UNI/Cognitive Computing/Thesis/Algonauts_Data"
    preds1_per_subject = []
    preds2_per_subject = []
    y_true_per_subject = []
    for subject_id in subjects:
        runconfig.subject_id = subject_id
        data = load_subject_data(runconfig, modality="all", area=None)

        pred_dict1 = jnp.load( os.path.join(f"{root_dir}", f"outputs_{framework1}_{modeltype1}_{hyper1}_{dataset}", f"s{subject_id}", "preds_test_sampling.npy"), allow_pickle= True).item()
        pred_dict2 = jnp.load(os.path.join(f"{root_dir}", f"outputs_{framework2}_{modeltype2}_{hyper2}_{dataset}", f"s{subject_id}", "preds_test_sampling.npy"), allow_pickle= True).item()

        y_true = []
        preds1 = []
        preds2 = []
        for i, ((epi, val), (_, val2)) in enumerate(zip(pred_dict1.items(), pred_dict2.items())):
            preds1.append(val[5:-5, :])
            preds2.append(val2[5:-5, :])
            y_true.append(data["test_sampling"]["fmri"][epi][5:-5, :])

        preds1 = jnp.concatenate(preds1, axis=0)
        preds2 = jnp.concatenate(preds2, axis =0)
        y_true = jnp.concatenate(y_true, axis=0)
        preds1_per_subject.append(np.array(preds1))
        preds2_per_subject.append(np.array(preds2))
        y_true_per_subject.append(np.array(y_true))
    return y_true_per_subject, preds1_per_subject, preds2_per_subject

def get_preds(root_dir, subjects, framework1, framework2, modeltype1, modeltype2, dataset, hyper1, hyper2):
    """
    Load two sets of model predictions for multiple subjects and concatenate episodes.
    """
    preds1_per_subject = []
    preds2_per_subject = []
    for subject_id in subjects:
        pred_dict1 = jnp.load( os.path.join(f"{root_dir}", f"outputs_{framework1}_{modeltype1}_{hyper1}_{dataset}", f"s{subject_id}", "preds_test_sampling.npy"), allow_pickle= True).item()
        pred_dict2 = jnp.load(os.path.join(f"{root_dir}", f"outputs_{framework2}_{modeltype2}_{hyper2}_{dataset}", f"s{subject_id}", "preds_test_sampling.npy"), allow_pickle= True).item()
        preds1 = []
        preds2 = []
        for i, ((epi, val), (_, val2)) in enumerate(zip(pred_dict1.items(), pred_dict2.items())):
            preds1.append(val[5:-5, :])
            preds2.append(val2[5:-5, :])

        preds1 = jnp.concatenate(preds1, axis=0)
        preds2 = jnp.concatenate(preds2, axis =0)
        preds1_per_subject.append(np.array(preds1))
        preds2_per_subject.append(np.array(preds2))
    return  preds1_per_subject, preds2_per_subject

def get_Y(root_dir, subjects, framework1, modeltype1, dataset, hyper1):
    """
    Load ground-truth fMRI time series (concatenated across episodes) for multiple subjects.

    Uses the episode keys from a loaded prediction dict to ensure the same episodes/order.
    """
    runconfig = rc.BaseRunConfigs(None, None)
    # runconfig.root_dir = "C:/Users/rogal/Documents/UNI/Cognitive Computing/Thesis/Algonauts_Data"
    y_true_per_subject = []
    # get correct episodes
    for subject_id in subjects:
        runconfig.subject_id = subject_id
        data = load_subject_data(runconfig, modality="all", area=None)
        pred_dict = jnp.load( os.path.join(f"{root_dir}", f"outputs_{framework1}_{modeltype1}_{hyper1}_{dataset}", f"s{subject_id}", "preds_test_sampling.npy"), allow_pickle= True).item()


        y_true = []

        for i, (epi, val) in enumerate(pred_dict.items()):
            y_true.append(data["test_sampling"]["fmri"][epi][5:-5, :])

        y_true = jnp.concatenate(y_true, axis=0)
        y_true_per_subject.append(np.array(y_true))
    return y_true_per_subject

def compute_voxelwise_r_per_subject_masked(Y, Y_pred, M):
    """
    Compute voxel/parcel-wise masked Pearson r per subject.
    """
    r_subj_voxel = jax.vmap(pearsonr_cols, in_axes=(0, 0, 0))(Y, Y_pred, M)
    return r_subj_voxel  

def group_mean_voxelwise_r(Y, Y_pred, M):
    """
    Compute group-mean voxel/parcel-wise correlation across subjects.
    """
    r_subj_voxel = compute_voxelwise_r_per_subject_masked(Y, Y_pred, M)  
    return jnp.mean(r_subj_voxel, axis=0) 

def compute_voxelwise_diff_map(Y, A, B, M):
    """
    Compute observed voxel/parcel-wise difference map between two prediction sets.
    """
    return group_mean_voxelwise_r(Y, A, M) - group_mean_voxelwise_r(Y, B, M)  

def jax_perm_diff_map_masked(Y, A, B, M, block_len: int, n_perm: int, seed: int = 0):
    """
    Permutation test for a voxel/parcel-wise difference map between two models.

    Observed map:
      group_mean_r(Y, A) - group_mean_r(Y, B)    (per voxel/parcel)

    Null maps:
      For each permutation:
        1) circularly shift Y, A, B, and M by a random amount
        2) swap A and B within random time blocks (Bernoulli per block)
        3) recompute the difference map

    P-values:
      Two-sided p-value per voxel using absolute values of null maps.

    Parameters
    ----------
    Y : jax.Array, shape (S, T, P)
        Ground truth.

    A, B : jax.Array, shape (S, T, P)
        Two prediction sets to compare.

    M : jax.Array, shape (S, T)
        Boolean mask.

    block_len : int
        Block length for swap mask.

    n_perm : int
        Number of permutations.

    seed : int, default=0
        PRNG seed.

    Returns
    -------
    obs_map : jax.Array, shape (P,)
        Observed voxel/parcel-wise difference map.

    p_map : jax.Array, shape (P,)
        Two-sided p-value per voxel/parcel.

    null_map : jax.Array, shape (n_perm, P)
        Null distribution maps.
    """
    S, T, P = Y.shape

    obs_map = compute_voxelwise_diff_map(Y, A, B, M) 

    key = jax.random.PRNGKey(seed)
    key_shifts, key_masks = jax.random.split(key)
    shifts = build_circular_shifts(key_shifts, n_perm, T)            
    masks  = build_block_swap_masks(key_masks, n_perm, T, block_len) 

    def one_perm_correct(shift, swap_mask_time):
        Ys = roll_time(Y, int(shift))
        As = roll_time(A, int(shift))
        Bs = roll_time(B, int(shift))
        Ms = roll_time(M, int(shift))
        sm = swap_mask_time[None, :, None]
        Ap = jnp.where(sm, Bs, As)
        Bp = jnp.where(sm, As, Bs)
        return group_mean_voxelwise_r(Ys, Ap, Ms) - group_mean_voxelwise_r(Ys, Bp, Ms)  # (P,)
    null_list = []
    for i, (shift, sm) in enumerate(zip(shifts, masks)):
        null_list.append(one_perm_correct(shift, sm))
        if i % 1000 == 0:
            logging.info(i)
    null_map = jnp.stack(null_list, axis=0) 

    abs_obs = jnp.abs(obs_map)[None, :]               
    abs_null = jnp.abs(null_map)                     
    # p-value per voxel (two-sided)
    p_map = (jnp.sum(abs_null >= abs_obs, axis=0) + 1.0) / (n_perm + 1.0)  

    return obs_map, p_map, null_map

def build_block_perm_indices(T: int, block_len: int, key):
    """
    Build a permutation index that shuffles time in contiguous blocks, preserving
    within-block order.
    """
    block_id = jnp.arange(T) // block_len                       
    n_blocks = int(jnp.max(block_id)) + 1                      
    within_block = jnp.arange(T) % block_len                    

    # random permutation of block ids
    perm_blocks = jax.random.permutation(key, jnp.arange(n_blocks)) 
    rank = jnp.empty_like(perm_blocks).at[perm_blocks].set(jnp.arange(n_blocks))  # (n_blocks,)

    order_key = rank[block_id] * block_len + within_block               

    idx = jnp.argsort(order_key)
    return idx.astype(jnp.int32)

def jax_perm_perf_vs_chance_blocked(
    Y, Y_pred, M, block_len: int, n_perm: int, seed: int = 0, two_sided: bool = False
):
    """
    Permutation test of performance vs chance by block-permuting predicted time order.

    Observed map:
      group_mean_r(Y, Y_pred)   (per voxel/parcel)

    Null maps:
      For each permutation:
        - block-permute the time dimension of Y_pred (shuffling blocks)
        - compute group_mean_r(Y, Y_pred_perm)

    P-values:
      One-sided by default (null >= observed). Two-sided if `two_sided=True`.

    Parameters
    ----------
    Y : jax.Array, shape (S, T, P)
        Ground truth.

    Y_pred : jax.Array, shape (S, T, P)
        Predictions.

    M : jax.Array, shape (S, T)
        Boolean mask.

    block_len : int
        Block length used for block-permutation.

    n_perm : int
        Number of permutations.

    seed : int, default=0
        PRNG seed.

    two_sided : bool, default=False
        Whether to compute two-sided p-values.

    Returns
    -------
    obs_map : jax.Array, shape (P,)
        Observed voxel/parcel-wise performance map.

    p_map : jax.Array, shape (P,)
        P-values per voxel/parcel.

    null_map : jax.Array, shape (n_perm, P)
        Null distribution maps.
    """
    S, T, P = Y.shape

    obs_map = group_mean_voxelwise_r(Y, Y_pred, M)  

    key = jax.random.PRNGKey(seed)
    perm_keys = jax.random.split(key, n_perm)

    # @jax.jit
    def one_perm(pk):
        idx = build_block_perm_indices(T, block_len, pk)        
        Y_pred_perm = Y_pred[:, idx, :]                       
        return group_mean_voxelwise_r(Y, Y_pred_perm, M)        

    null_map = jnp.stack([one_perm(pk) for pk in perm_keys], axis=0)  

    if two_sided:
        thresh = jnp.abs(obs_map)[None, :]
        p_map = (jnp.sum(jnp.abs(null_map) >= thresh, axis=0) + 1.0) / (n_perm + 1.0)
    else:
        thresh = obs_map[None, :]
        p_map = (jnp.sum(null_map >= thresh, axis=0) + 1.0) / (n_perm + 1.0)

    return obs_map, p_map, null_map


def main():
    parser = argparse.ArgumentParser(description="Run block permutation test for model comparison.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., hyperset)")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 5], help="List of subject IDs")
    parser.add_argument("--n_perm", type=int, default=1000, help="Number of permutations")
    parser.add_argument("--results_dir", type=str, required=True, help="Root directory of the data")
    parser.add_argument("--parcellevel",action="store_true", help="Enable parcel level mode")
    parser.add_argument("--subjectlevel",action="store_true", help="Enable subject level mode")
    args = parser.parse_args()

    dataset = args.dataset
    results_dir = args.results_dir
    subjects = args.subjects
    n_perm = args.n_perm
    parcellevel = args.parcellevel
    subjectlevel = args.subjeclevel
    chancecomp = args.chancecomp
    brainarea = args.brainarea
    area_ids = np.load(f"{results_dir}/area_ids.npy")

    comparisons = [
        # (("sfm", "gru", "default"), ("sfm", "lstm", "default")),
        # (("sfm", "gru", "default"), ("sfm", "simple", "default")),
        # (("sfm", "gru", "default"), ("sfm", "tcn", "default")),
        # (("sfm", "lstm", "default"), ("sfm", "tcn", "default")),
        # (("sfm", "lstm", "default"), ("sfm", "simple", "default")),
        # (("sfm", "tcn", "default"), ("sfm", "simple", "default")),

        # AFM hyper
        # (("afm", "gru", "default"), ("afm", "lstm", "default")),
        # (("afm", "gru", "default"), ("afm", "simple", "default")),
        # # (("afm", "gru", "default"), ("afm", "tcn", "default")),
        # # (("afm", "lstm", "default"), ("afm", "tcn", "default")),
        # (("afm", "lstm", "default"), ("afm", "simple", "default")),
        # (("afm", "tcn", "default"), ("afm", "simple", "default")),

        # Default full testset
        #SFM
        # (("sfm", "gru", "default"), ("sfm", "lstm", "default")),
        # (("sfm", "gru", "default"), ("sfm", "simple", "default")),
        # (("sfm", "gru", "default"), ("sfm", "tcn", "default")),
        # (("sfm", "lstm", "default"), ("sfm", "tcn", "default")),
        # (("sfm", "lstm", "default"), ("sfm", "simple", "default")),
        # (("sfm", "tcn", "default"), ("sfm", "simple", "default")),

        # # AFM
        # # (("afm", "gru", "default"), ("afm", "lstm", "default")),
        # (("afm", "gru", "default"), ("afm", "simple", "default")),
        # (("afm", "gru", "default"), ("afm", "tcn", "default")),
        # (("afm", "lstm", "default"), ("afm", "tcn", "default")),
        # (("afm", "lstm", "default"), ("afm", "simple", "default")),
        # (("afm", "tcn", "default"), ("afm", "simple", "default")),

        # # LSNDE
        (("latentsde", "gru", "default"), ("latentsde", "lstm", "default")),
        (("latentsde", "gru", "default"), ("latentsde", "simple", "default")),
        (("latentsde", "gru", "default"), ("latentsde", "tcn", "default")),
        (("latentsde", "lstm", "default"), ("latentsde", "tcn", "default")),
        (("latentsde", "lstm", "default"), ("latentsde", "simple", "default")),
        (("latentsde", "tcn", "default"), ("latentsde", "simple", "default")),

        # # combi
        # (("sfm", "gru", "default"), ("afm", "gru", "default")),
        # # (("sfm", "lstm", "default"), ("afm", "lstm", "default")),
        # (("sfm", "simple", "default"), ("afm", "simple", "default")),
        # (("sfm", "tcn", "default"), ("afm", "tcn", "default")),

        # regr
        # (("sfm", "gru", "default"), ("regression", "None", "default")),
        # (("sfm", "lstm", "default"), ("regression", "None", "default")),
        # (("sfm", "tcn", "default"), ("regression", "None", "default")),
        # (("sfm", "simple", "default"), ("regression", "None", "default")),
        # (("afm", "gru", "default"), ("regression", "None", "default")),
        # (("afm", "lstm", "default"), ("regression", "None", "default")),
        # (("afm", "tcn", "default"), ("regression", "None", "default")),
        # (("afm", "simple", "default"), ("regression", "None", "default")),
        # (("sfm", "lstm", "default"), ("afm", "lstm", "default")),
        # (("sfm", "simple", "default"), ("afm", "simple", "default")),
        # (("sfm", "tcn", "default"), ("afm", "tcn", "default")),

        #comb vs indiv
        # (("afm", "gru", "optimized"), ("afm", "gru", "optimized")),
        # (("regression", "None", "default"), ("regression", "None", "default")),
        # (("sfm", "gru", "optimized"), ("sfm", "gru", "optimized")),

        # (("afm", "gru", "optimized"), ("sfm", "gru", "optimized")),
        # (("regression", "None", "default"), ("sfm", "gru", "optimized")),
        # (("sfm", "gru", "optimized"), ("afm", "gru", "optimized")),

        # (("afm", "gru", "optimized"), ("sfm", "gru", "optimized")),
        # (("regression", "None", "default"), ("sfm", "gru", "optimized")),
        # (("sfm", "gru", "optimized"), ("afm", "gru", "optimized")),

    ]
    comparisons_results = {}
    y_true_per_subject = get_Y(results_dir, subjects, comparisons[0][0][0], comparisons[0][0][1], dataset, comparisons[0][0][2])
    Y_pad, M = pad_and_make_mask(y_true_per_subject)
    # loaded_data = np.load("/content/drive/MyDrive/y_true_per_subject.npz")
    # Y_pad, M =loaded_data["Y_pad"], loaded_data["M"]

    Y = jax.device_put(Y_pad)
    M = jax.device_put(M)
    os.makedirs("Results", exist_ok=True)
    del Y_pad
    for ((framework1, modeltype1, hyper1), (framework2, modeltype2, hyper2)) in comparisons:
        # Load data
        preds1_per_subject, preds2_per_subject = get_preds(
            results_dir,
            subjects,
            framework1,
            framework2,
            modeltype1,
            modeltype2,
            dataset,
            hyper1,
            hyper2
        )

        A_pad, _ = pad_and_make_mask(preds1_per_subject)
        B_pad, _ = pad_and_make_mask(preds2_per_subject)
        del preds1_per_subject, preds2_per_subject

        A = jax.device_put(A_pad)
        B = jax.device_put(B_pad)
        del A_pad, B_pad

        block_len = 22
        logging.info("Preds loaded")

        if parcellevel:
            if chancecomp:
                print("comparing to chance")
                for s, subject_id in enumerate(subjects):
                    obs_map, p_map, null_map = jax_perm_perf_vs_chance_blocked(
                        Y[[s], :], A[[s], :], M[[s], :],
                        block_len=block_len, n_perm=n_perm, seed=42
                    )
                    _ = jax.block_until_ready(null_map)
                    comparisons_results[f"{subject_id}_parcels_vs_chance"] = (obs_map, p_map)

                    if brainarea:
                        obs_area, p_area, null_area = perf_vs_chance_area_from_maps(
                            obs_map, null_map, area_ids, two_sided=False
                        )
                        _ = jax.block_until_ready(null_area)
                        comparisons_results[f"{subject_id}_areas_vs_chance"] = (obs_area, p_area)

                # group mean across subjects
                obs_map, p_map, null_map = jax_perm_perf_vs_chance_blocked(
                    Y, A, M, block_len=block_len, n_perm=n_perm, seed=42
                )
                _ = jax.block_until_ready(null_map)
                comparisons_results["mean_parcels_vs_chance"] = (obs_map, p_map)

                if brainarea:
                    obs_area, p_area, null_area = perf_vs_chance_area_from_maps(
                                obs_map, null_map, area_ids, two_sided=False
                            )
                    _ = jax.block_until_ready(null_map)
                    comparisons_results["mean_areas_vs_chance"] = (obs_map, p_map)

            else:
                for s, subject_id in enumerate(subjects):
                    obs_map, p_map, null_map = jax_perm_diff_map_masked(
                        Y[[s], :], A[[s], :],  B[[s], :], M[[s], :], block_len=block_len, n_perm=n_perm, seed=42
                    )
                    _ = jax.block_until_ready(null_map)
                    key = f"{framework1}_{modeltype1}_{hyper1}__vs__{framework2}_{modeltype2}_{hyper2}"
                    comparisons_results[f"{subject_id}_" +key + "_parcels"] = (obs_map, p_map)

                    if brainarea:
                        obs_area, p_area, null_area = perf_vs_chance_area_from_maps(
                                    obs_map, null_map, area_ids, two_sided=True
                                )
                        _ = jax.block_until_ready(null_area)
                        comparisons_results[f"{subject_id}_"  + key + "_areas"] = (obs_area, p_area)

                #mean
                obs_map, p_map, null_map = jax_perm_diff_map_masked(
                    Y, A, B, M, block_len=block_len, n_perm=n_perm, seed=42
                )
                _ = jax.block_until_ready(null_map)
                key = f"{framework1}_{modeltype1}_{hyper1}__vs__{framework2}_{modeltype2}_{hyper2}"
                comparisons_results["mean_" +key + "_parcels"] = (obs_map, p_map)

                if brainarea:
                    obs_area, p_area, null_area = perf_vs_chance_area_from_maps(
                                obs_map, null_map, area_ids, two_sided=True
                            )
                    _ = jax.block_until_ready(null_area)
                    comparisons_results["mean_" + key + "_areas"] = (obs_area, p_area)

        else:
            obs, p, null = jax_perm_diff_of_means_masked(
                Y, A, B, M, block_len=block_len, n_perm=n_perm, seed=42, subject_level=subjectlevel
            )
            _ = jax.block_until_ready(null)
            key = f"{framework1}_{modeltype1}_{hyper1}__vs__{framework2}_{modeltype2}_{hyper2}"
            comparisons_results[key] = (obs, p)
            if subjectlevel:
                for i, sub in enumerate(subjects):
                    png_name = os.path.join("Results",f"perm_null_{framework1}_{modeltype1}_{framework2}_{modeltype2}_{dataset}_s{sub}.png")
                    save_path = save_null_plot(null[:, i], obs[i], p[i], out_path=png_name, bins=50)
                    logging.info(f"Saved plot to {save_path}")
            else:
                png_name = os.path.join("Results",f"perm_null_{framework1}_{modeltype1}_{framework2}_{modeltype2}_{dataset}.png")
                save_path = save_null_plot(null, obs, p, out_path=png_name, bins=50)
                logging.info(f"Saved plot to {save_path}")
    del A, B

    # Save in one file for later FDR correction
    outfile = os.path.join("Results", f"{framework1}to{framework2}.npz")
    np.savez(outfile, **comparisons_results)

if __name__ == "__main__":
    main()