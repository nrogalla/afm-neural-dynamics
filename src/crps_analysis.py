import os
import pandas as pd
import jax.numpy as jnp
import numpy as np
from data.dataloaders import load_subject_data
import utils.runconfigs as rc

import jax
import logging
logging.basicConfig(level=logging.INFO)

root_data_dir = ""
subjects = [1,2,3,5]

@jax.jit
def crps_ensemble_jax(y, x):
    """
    y: shape (T,)      observations
    x: shape (T, S)    ensemble members
    returns: shape (T,) CRPS per time
    """
    term1 = jnp.mean(jnp.abs(x - y[:, None]), axis=1)

    # pairwise absolute differences along ensemble dimension
    diffs = jnp.abs(x[:, :, None] - x[:, None, :])  # (T, S, S)
    term2 = 0.5 * jnp.mean(diffs, axis=(1, 2))

    return term1 - term2

def compute_crps(pred, obs):
    """
    pred: array (T, S, P)
    obs : array (T, P)

    Returns:
        crps_per_t: shape (T, P)
        crps_avg: scalar
    """
    crps_per_t = []
    for parcel in range(pred.shape[2]):
      crps = np.mean(crps_ensemble_jax(obs[:, parcel], pred[:,:, parcel]))
      crps_per_t.append(crps)  # shape (T, P)

    # Average over the forecast horizon
    crps_avg = np.mean(crps_per_t)

    return crps_per_t, crps_avg

res_dir = r"outputs"

frameworks   = ["afm", "sfm"]
subject_ids  = [1, 2, 3, 5]

results = []  # will store dicts: {"framework": f, "subject": s, "crps": value}


for subject_id in subject_ids:
    runconfig = rc.BaseRunConfigs(None, None)
    runconfig.root_dir = root_data_dir
    runconfig.subject_id = subject_id  # <--- use the loop subject

    data = load_subject_data(runconfig, modality="all", area=None)
    Ys_test = data["test_sampling"]["fmri"]

    
    for framework in frameworks:
        try:
            prob_series_dict = np.load(
                os.path.join(f"{res_dir}", f"outputs_{framework}_gru_s{subject_id}", "preds_test_sampling_samples.npy"),
                allow_pickle=True
            ).item()

            prob_series = np.concatenate(list(prob_series_dict.values()), axis=0)
            true_series = np.concatenate(
                [Ys_test[item] for item in Ys_test if item in prob_series_dict],
                axis=0
            )

            crps_per_parcel, crps_avg = compute_crps(prob_series, true_series)
            logging.info(f"{framework} s{subject_id}: {crps_avg}")


            # store a row
            results.append({
                "framework": framework,
                "subject": subject_id,
                "crps": crps_avg
            })
            file_path = os.path.join(f"crps_outputs_{framework}", f"{subject_id}")

            os.makedirs(file_path, exist_ok=True)
            np.save(file_path, crps_per_parcel, allow_pickle=True)

            del prob_series_dict, prob_series, true_series
        except:
            continue

df = pd.DataFrame(results)

crps_table = df.pivot(index="subject", columns="framework", values="crps")
logging.info(crps_table)
