import numpy as np
from scipy.stats import pearsonr
from nilearn import plotting
import seaborn as sns
from nilearn.maskers import NiftiLabelsMasker
import os
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from data.dataloaders import load_subject_data
from evaluation.evaluation import compute_pearson_score_full
import utils.runconfigs as rc

import utils.utils_combi as utils
import h5py
import jax
from matplotlib.gridspec import GridSpec
from evaluation.permutation_test_mutiple_comparisons import reduce_parcel_matrix_to_areas, reduce_parcel_vector_to_areas


palette =  sns.color_palette("Set3")[6:7] +sns.color_palette("Set3")[3:5] + sns.color_palette("Set3")[9:10]
###
# Helpers
###
def get_correlation_parcels(fmri_test, fmri_test_pred):
    encoding_accuracy = np.zeros((fmri_test.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_test[:, p],
            fmri_test_pred[:, p])[0]
    return encoding_accuracy


def get_noise_ceiling(root_data_dir, subjects, level, area_ids = None):

    corr_per_subject = []
    for subject in subjects:
        fmri = {}
        fmri_run_1_list = []
        fmri_run_2_list = []
        ### Load the fMRI responses for Movie10 ###
        # Data directory
        fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
        fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
            'fmri', f'sub-0{subject}', 'func', fmri_file)
        # Load the the fMRI responses
        fmri_movie10 = h5py.File(fmri_dir, 'r')
        fmri_movie10 = h5py.File(fmri_dir, 'r')
        for key, val in fmri_movie10.items():
            fmri[key[13:]] = val[:].astype(np.float32)
        fmri = dict(sorted(fmri.items(), key=lambda item: utils.sort_key(item[0])))

        for key, val in fmri.items():
            
            if "run-1" in key:
                fmri_run_1_list.append(val[:].astype(np.float32))
            elif "run-2" in key:
                fmri_run_2_list.append(val[:].astype(np.float32))
        fmri_run_1 = np.concatenate(fmri_run_1_list, axis = 0)
        
        fmri_run_2 = np.concatenate(fmri_run_2_list, axis = 0)

        CC_self  = get_correlation_parcels(fmri_run_1, fmri_run_2)
        corr_per_subject.append(np.round(CC_self, 3))
        del fmri_movie10
        del fmri

    if level == 'per_subject':
        corr_per_subject = [np.mean(corr_per_subject, axis=1)]
    elif level == 'across_subjects':
        corr_per_subject = [np.mean(np.mean(corr_per_subject, axis=1), axis=0)]
    elif level == 'per_parcel_across_subjects':
        corr_per_subject = [np.mean(corr_per_subject, axis=0)]
    elif level == "per_area_per_subjects":
        corr_per_subject = reduce_parcel_matrix_to_areas(np.array(corr_per_subject), area_ids)
    elif level == "per_area_across_subjects":
        corr_per_subject = np.mean(reduce_parcel_matrix_to_areas(np.array(corr_per_subject), area_ids), axis = 0)
    noise_ceilings = []
    for CC_self in corr_per_subject:
        noise_ceilings.append(np.sqrt(2 / (1 + (1 / CC_self))) )
    return noise_ceilings

def get_preds(root_dir, subjects, framework1, modeltype1, dataset, hyper1, type = "test", group = "individual"):
    preds1_per_subject = []
    for subject_id in subjects:
        pred_dict1 = jnp.load( os.path.join(f"{root_dir}", f"outputs_{framework1}_{modeltype1}_{hyper1}_{dataset}_{group}", f"s{subject_id}", f"preds_{type}_sampling.npy"), allow_pickle= True).item()
        preds1 = []
        for i, (epi, val) in enumerate(pred_dict1.items()):
            preds1.append(val[5:-5, :])
        preds1 = jnp.concatenate(preds1, axis=0)
        preds1_per_subject.append(np.array(preds1))
    return  preds1_per_subject

def get_Y(root_dir, subjects, framework1, modeltype1, dataset, hyper1, type="test", group = "individual"):
    runconfig = rc.BaseRunConfigs(None, None)
    runconfig.root_dir = "C:/Users/rogal/Documents/UNI/Cognitive Computing/Thesis/Algonauts_Data"
    y_true_per_subject = []
    # get correct episodes
    for subject_id in subjects:
        runconfig.subject_id = subject_id
        data = load_subject_data(runconfig, modality="all", area=None)
        pred_dict = jnp.load( os.path.join(f"{root_dir}", f"outputs_{framework1}_{modeltype1}_{hyper1}_{dataset}_{group}", f"s{subject_id}", f"preds_{type}_sampling.npy"), allow_pickle= True).item()
     

        y_true = []
       
        for i, (epi, val) in enumerate(pred_dict.items()):
            y_true.append(data["test_sampling"]["fmri"][epi][5:-5, :])

        y_true = jnp.concatenate(y_true, axis=0)
        y_true_per_subject.append(np.array(y_true))
        del data, y_true
    return y_true_per_subject
   
def add_signif_lines(ax, right_val,  max_height, sign_dict, f):
    comparisons = []
    for (x, y, col) in zip(
        np.arange(right_val, right_val-0.7, -0.2),
        np.arange(max_height, max_height-0.07, -0.03),
        palette[::-1]
    ):
        comparisons.append((y, right_val-0.2, x, col))

    # keep track of already-checked unordered pairs
    seen_pairs = set()

    for ((y, i0, i1, col), m1) in zip(comparisons, ["tcn", "gru", "lstm"]):
        # horizontal line (once per m1)
        ax.hlines(y, i0 - 0.4, i1, color="black", lw=1)

        for (xi, m2) in zip(np.arange(-0.3, i1, 0.2), ["simple", "lstm", "gru"]):
            # unordered pair, so ("tcn","lstm") == ("lstm","tcn")
            pair = tuple(sorted((m1, m2)))

            # skip identical (gru,gru) etc. and already-done pairs
            if m1 == m2 or pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            # look up p-value (either order in dict)
            try:
                p = sign_dict[f"{f}_{m1}_{m2}"]
            except KeyError:
                p = sign_dict[f"{f}_{m2}_{m1}"]

            if p <= 1.00e-04:
                marker = "****"
            elif p <= 1.00e-03:
                marker = "***"
            elif p <= 1.00e-02:
                marker = "**"
            elif p <= 5.00e-02:
                marker = "*"
            else:
                continue

            
            ax.text(right_val -0.3 + xi, y+0.012, marker, color=col,
                    ha='center', va='top', fontsize=18, weight= "bold")
            
def add_values_above_bars(ax, x_pos_adjust=0.1, y_pos_adjust=0.004):

    ax.tick_params(axis='x', pad=15)
    # Add value labels above bars
    for p in ax.patches:
        height = p.get_height()
        if height != 0:

            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2. -x_pos_adjust , height -y_pos_adjust),
                        ha='center', va='bottom', fontsize=7, color='black',
                        xytext=(0, 5), textcoords='offset points')

def get_encoder_comparison_figure(root_data_dir, subjects,df):
    df["NC"] = get_noise_ceiling(root_data_dir, subjects, "per_subject")[0] 

    cols_to_divide = df.columns.difference(["Subject", "NC"])

    df_normalized = df.copy()
    df_normalized[cols_to_divide] = np.round(df[cols_to_divide].div(df["NC"], axis=0), 3)

    df_long = df_normalized.melt(
        id_vars='Subject',
        var_name='Framework_Model',
        value_name='Correlation'
    )

    tmp = df_long['Framework_Model'].str.split('_', n=1, expand=True)
    df_long['Framework'] = tmp[0]

    df_long['Model'] = tmp[1].fillna(df_long['Framework'])

    df_long = df_long[['Subject', 'Framework', 'Model', 'Correlation']]

    plt.figure(figsize=(8, 3.5))
    order = ['SFM', 'AFM']
    hue_order = ['sRNN', 'LSTM', 'GRU', 'TCN']

    df_hued = df_long[df_long['Framework'].isin(['SFM', 'AFM'])]

    ax = sns.barplot(
        data=df_hued,
        x='Framework', y='Correlation',
        hue='Model',
        order=order,         
        hue_order=hue_order,
        errorbar="sd",
        palette=palette,
        edgecolor='black'
    )

    # Add value labels above bars
    add_values_above_bars(ax,0.05, 0.01)
    ax.tick_params(axis='x', pad=17)
    plt.ylabel("r*", labelpad=15, fontsize = 14)
    # plt.xlabel("framework", labelpad=15, fontsize = 14)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title='encoder',
            bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
    return ax, df_hued

def compute_encoding_accuracy_imgs(
    root_data_dir,
    fmri_val,
    fmri_val_pred,
    subjects,
    reject_maps=None,           
    include_mean=True,
    ob_map = None
):
    """
    Build glass-brain images for each subject, plus mean panel.
    Parameters
    ----------
    root_data_dir : str
    fmri_val, fmri_val_pred : list-like [n_subjects] of arrays shape (T, P)
    subjects : list[int]
    reject_maps : None or list-like [n_subjects] of bool arrays/lists length P (True = include)
    include_mean : bool

    Returns
    -------
    imgs   : list of Niimg-like (mean first if include_mean=True)
    titles : list of str (aligned with imgs)
    """
    if len(subjects) == 0:
        raise ValueError("`subjects` must be non-empty.")

    using_masks = reject_maps is not None
    if reject_maps is not None:
        reject_maps = [np.asarray(m, dtype=bool) for m in reject_maps]

    ref_subject = subjects[0]
    ref_atlas_file = (
        f"sub-0{ref_subject}_space-MNI152NLin2009cAsym_"
        "atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
    )
    ref_atlas_path = os.path.join(
        root_data_dir, "algonauts_2025.competitors", "fmri",
        f"sub-0{ref_subject}", "atlas", ref_atlas_file
    )
    ref_masker = NiftiLabelsMasker(labels_img=ref_atlas_path).fit()

    imgs, titles = [], []
    acc_list = []

    for i, subject in enumerate(subjects):
        if ob_map is None:
            y_true = fmri_val[i]
            y_pred = fmri_val_pred[i]
            n_parc = y_true.shape[1]

            if using_masks:
                if reject_maps[i].shape[0] != n_parc:
                    raise ValueError(
                        f"reject_maps[{i}] has length {reject_maps[i].shape[0]} but subject {subject} has {n_parc} parcels."
                    )

            acc = np.empty(n_parc, dtype=np.float32)
            for p in range(n_parc):
                a = y_true[:, p]
                b = y_pred[:, p]
                if (np.std(a) == 0) or (np.std(b) == 0):
                    acc[p] = np.nan
                else:
                    acc[p] = pearsonr(a, b)[0]
        else:
            acc = ob_map[i]
            n_parc = ob_map[i].shape[0]
        acc_list.append(acc)

        # Apply mask: keep True, hide False as -3 for later handling
        if using_masks:
            mask = reject_maps[i]
            acc = np.where(mask, acc, -3)

    
        atlas_file = (
            f"sub-0{subject}_space-MNI152NLin2009cAsym_"
            "atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
        )
        atlas_path = os.path.join(
            root_data_dir, "algonauts_2025.competitors", "fmri",
            f"sub-0{subject}", "atlas", atlas_file
        )
        masker = NiftiLabelsMasker(labels_img=atlas_path).fit()
        print(f"Subject {subject}, Min: {np.nanmin(np.where(np.asarray(acc) == -3, np.nan, acc))}, Mean: {np.nanmean(np.where(np.asarray(acc) == -3, np.nan, acc))}, Max: {np.nanmax(acc)}")
        acc_nii = masker.inverse_transform(acc)

        imgs.append(acc_nii)
        if using_masks:
            n_kept = int(np.sum(mask))
            titles.append(f"sub-0{subject} ({n_kept}/{n_parc})")
        else:
            titles.append(f"sub-0{subject}")

    if include_mean:
        mean_acc = np.mean(np.array(acc_list), axis=0)
        if using_masks:
            mask = reject_maps[i+1]
            mean_acc = np.where(mask, mean_acc, -3)
            
        print(f"Mean, Min: {np.nanmin(np.where(np.asarray(mean_acc) == -3, np.nan, mean_acc))}, Mean: {np.nanmean(np.where(np.asarray(mean_acc) == -3, np.nan, mean_acc))}, Max: {np.nanmax(mean_acc)}")
        mean_nii = ref_masker.inverse_transform(mean_acc)

        imgs = [mean_nii] + imgs
        
        if using_masks:
            n_kept_mean = int(np.sum(mask))
            titles = [f"mean ({n_kept_mean}/{mean_acc.size})"] + titles
        else:
            titles = [f"mean"] + titles

    return imgs, titles

def stack_glass_brains(
    nii_list,
    titles,
    out_png,
    cmap='hot',
    display_mode='lyrz',
    vmin=None,
    vmax=None,
    label="$r$",
    cbar_rel_width=0.035,
    cbar_pad=0.02
):
    if vmin is None or vmax is None:
        mins, maxs = [], []
        for img in nii_list:
            d = np.asarray(img.get_fdata(), dtype=float)
            mins.append(np.nanmin(np.where(np.asarray(d) == -3, np.nan, d)))
            maxs.append(np.nanmax(np.where(np.asarray(d) == -3, np.nan, d)))
        if vmin is None:
            vmin = float(np.min(mins))
        if vmax is None:
            vmax = float(np.max(maxs))

    n = len(nii_list)

    fig = plt.figure(figsize=(10, 3.2 * n), constrained_layout=True)
    gs = GridSpec(nrows=n, ncols=3, figure=fig,
                  width_ratios=[1.0, cbar_rel_width, cbar_pad])
    title_positions = []
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_under(color='0.5')

    for i, img in enumerate(nii_list):
        ax = fig.add_subplot(gs[i, 0])
        ax.axis("off")
        disp = plotting.plot_glass_brain(
            img,
            display_mode=display_mode,
            cmap=cmap,
            colorbar=False,
            plot_abs=False,
            symmetric_cbar=False,
            vmin=vmin, vmax=vmax,
            figure=fig, axes=ax
        )
        
        pos = ax.get_position()
        title_y = pos.y1 + 0.01 
        title_positions.append((pos.x0 + pos.width/2, title_y))

    for (x, y), title in zip(title_positions, titles):
        fig.text(x, y, title, ha="center", va="bottom", fontsize=12)

    # Shared colorbar
    cax = fig.add_subplot(gs[2, 1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label, rotation=90, labelpad=10, fontsize=12)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def compute_encoding_accuracy(root_data_dir, fmri_val, fmri_val_pred, subject, modality, folder_path):
    """
    Compare the  recorded (ground truth) and predicted fMRI responses, using a
    Pearson's correlation. The comparison is perfomed independently for each
    fMRI parcel. The correlation results are then plotted on a glass brain.

    Parameters
    ----------
    fmri_val : float
        fMRI responses for the validation movies.
    fmri_val_pred : float
        Predicted fMRI responses for the validation movies
    subject : int
        Subject number used to train and validate the encoding model.
    modality : str
        Feature modality used to train and validate the encoding model.

    """

    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
            fmri_val_pred[:, p])[0]
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)

    ### Map the prediction accuracy onto a 3D brain atlas for plotting ###
    atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    atlas_path = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'atlas', atlas_file)
    atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
    atlas_masker.fit()
    encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)

    ### Plot the encoding accuracy ###
    title = f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: " + str(mean_encoding_accuracy)
    display = plotting.plot_glass_brain(
        encoding_accuracy_nii,
        display_mode="lyrz",
        cmap='hot_r',
        colorbar=True,
        plot_abs=False,
        symmetric_cbar=False,
        title=title
    )
    colorbar = display._cbar
    colorbar.set_label("$r$", rotation=90, labelpad=12, fontsize=12)
    display.savefig(f"{folder_path}/encoding_accuracy_sub-0{subject}_{modality}.png", dpi=300)

    return plotting.show()

def plot_time_series_comparison(res_dir, root_data_dir, start, stop, parcel, epi, palette, subject_id):

    glm_series_dict   = np.load(os.path.join(f"{res_dir}", "outputs_regression_None_default_testset_individual", f"s{subject_id}", "preds_test_sampling.npy"), allow_pickle=True).item()
    sfm_series_dict   = np.load(os.path.join(f"{res_dir}",  f"outputs_sfm_gru_s{subject_id}", "preds_test_sampling_samples.npy"), allow_pickle=True).item()
    afm_series_dict   = np.load(os.path.join(f"{res_dir}",  f"outputs_afm_gru_s{subject_id}", "preds_test_sampling_samples.npy"), allow_pickle=True).item()
    # # lsndefm_series = np.load(fr"{res_dir}\outputs_latentsde_gru_optimized_testset_individual\s1\preds_test_sampling.npy", allow_pickle=True).item()[epi][start:stop, parcel]
    afm_series =afm_series_dict[epi][start:stop, parcel]
    sfm_series =sfm_series_dict[epi][start:stop, parcel]
    glm_series =glm_series_dict[epi][start:stop, parcel]
    runconfig = rc.BaseRunConfigs(None, None)
    runconfig.root_dir = root_data_dir
    runconfig.subject_id = subject_id
    data = load_subject_data(runconfig, modality="all", area=None)
    Ys_test = data["test_sampling"]["fmri"]
    true_series = Ys_test[epi][start:stop, parcel]
    t = np.arange(start, stop)

    T = len(t)
    N_samples = afm_series.shape[1]   # second 100

    # --- deterministic series (1 value per t) ---
    df_det = pd.DataFrame({
        "t": np.tile(t, 2),
        "Series": np.repeat(["observed", "GLM"], T),
        "Value": np.concatenate([true_series, glm_series]),
    })

    # --- AFM: many samples per t ---
    df_sfm = pd.DataFrame({
        "t": np.repeat(t, N_samples),
        "Series": "SFM",
        "Value": sfm_series.reshape(-1),
    })
    df_afm = pd.DataFrame({
        "t": np.repeat(t, N_samples),
        "Series": "AFM",
        "Value": afm_series.reshape(-1),
    })

    # Combine
    df = pd.concat([df_det, df_sfm, df_afm], ignore_index=True)

    plt.figure(figsize=(12, 5))
    ax = sns.lineplot(
        data=df,
        x="t",
        y="Value",
        hue="Series",
        palette=palette,
        errorbar=("ci", 95)   # uses AFM's multiple samples to compute CI
    )

    ax.legend(
        title='framework',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )
    plt.ylabel("BOLD signal", labelpad=15)#, fontsize=14)
    plt.xlabel("frame", labelpad=15)#, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_timeseries_with_confidence(x_true, x_pred, channel=0, confidence=0.95, title=None, ax=None):
    """
    Plot time series with confidence intervals

    Args:
        x_true: True time series data (T,)
        x_samples: Sampled time series data (n_samples, T, channels)
        ts: Time axis
        channel: Channel to plot
        confidence: Confidence level (default: 0.90)
        title: Plot title (optional)
        ax: Matplotlib axis object for subplots
    """
    # Calculate statistics
    mean = jnp.mean(x_pred, axis=1)
 
    percentile_low = jnp.percentile(x_pred, (1 - confidence) * 100 / 2, axis=1)
    percentile_high = jnp.percentile(x_pred, (1 + confidence) * 100 / 2, axis=1)

    ts = np.arange(0, np.shape(x_true)[0] * 1.49, 1.49)
    # Plot on the provided axis
    lavender = "#6256e7"
    coral = "#c72f1e"

    ax.fill_between(
        ts, percentile_low, percentile_high,
        color="red", alpha=0.2, label=f'{confidence*100:.0f}% Confidence Interval'
    )

    ax.plot(ts, mean, color=coral, label='Prediction', linewidth=2)
    ax.plot(ts, x_true, color=lavender, label='True', linewidth=2)

    # Customize plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    if title:
        ax.set_title(title)
    # ax.legend()
    ax.grid(True, alpha=0.3)


def plot_multiple_channels(x_true, x_samples, channels, folder):
    """
    Plot multiple channels in a 3x2 grid

    Args:
        x_true: True time series data (T,)
        x_samples: Sampled time series data (n_samples, T, channels)
        ts: Time axis
        channels: List of channels to plot
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    axs = axs.flatten()

    for i, channel in enumerate(channels):
        plot_timeseries_with_confidence(
            x_true=x_true[:, channel],
            x_pred=x_samples[:, :,  channel],
            #ts=ts,
            channel=channel,
            confidence=0.95,
            title=f'Parcel {channel}',
            ax=axs[i]
        )
    fig.subplots_adjust(hspace=0.35, wspace=0.25, right=0.82)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
            handles, labels,
            loc='center left',           
            bbox_to_anchor=(1.02, 0.5),  
            borderaxespad=0
        )

    fig.tight_layout(pad=3.0)
    plt.savefig(f"{folder}/timeseries_plot.png", dpi=300) 
    plt.tight_layout()
    plt.show()

def plot_time_series():
    runconfig = rc.BaseRunConfigs(None, None)
    runconfig.root_dir = "/Users/nicolerogalla/Documents/AFM/Algonauts_Data"

    runconfig.subject_id = 1
    data = load_subject_data(runconfig, modality="all", area=None)
    Ys_test = data["test_sampling"]["fmri"]
    del data

    samp = np.load(r"C:\Users\rogal\Documents\UNI\Cognitive Computing\Thesis\testing_sde\src_sde\out\predstest_sampling_samples_sde.npy", allow_pickle= True).item()
    folder = r"C:\Users\rogal\Documents\UNI\Cognitive Computing\Thesis\testing_sde\src_sde\out"
    epi = "s06e01a"
    channels = list(np.random.randint(0,np.shape(samp[epi])[2], size = 6))

    plot_multiple_channels(Ys_test[epi][100:150], samp[epi][100:150], channels, folder)

def plot_temporal_ablation(root_data_dir, subjects, network_names, area_ids, palette, ablation_type = "context"):
    #setup
    res_dir = r"/Users/nicolerogalla/Documents/AFM/new_outputs"

    subject_id = 1

    runconfig = rc.BaseRunConfigs(None, None)

    runconfig.root_dir = root_data_dir
    runconfig.subject_id = subject_id 

    data = load_subject_data(runconfig, modality="all", area=None)
    Ys_test = data["test_sampling"]["fmri"]
    frameworks = ["sfm", "afm"]
    if ablation_type == "context":
        windows = np.arange(0, 55, 5)
    elif ablation_type == "prediction": 
        windows = np.arange(0, 30, 5)
    else:
        print("Ablation not implemented.")
        return
    whole_brain_rows = []
    rows = []
    noise_ceiling = get_noise_ceiling(root_data_dir, subjects, "per_area_per_subjects", area_ids)
    noise_ceilingwb = get_noise_ceiling(root_data_dir, subjects = subjects,level="per_subject")
    if ablation_type == "prediction":
        ablation= "segment"
    else: 
        ablation= ablation_type
    for window in windows:
        for framework in frameworks:
        # preds_window = f"/Users/nicolerogalla/Documents/AFM/new_outputs/outputs_afm_gru_s1_context_{int(window)}"
            try:

                prob_series_dict = np.load(
                    os.path.join(f"{res_dir}", f"outputs_{framework}_gru_s{subject_id}_{ablation}_{int(window)}", "preds_test_sampling.npy"),
                    allow_pickle=True
                ).item()
            except: 
                prob_series_dict = np.load(
                os.path.join(f"{res_dir}", f"outputs_{framework}_gru_s{subject_id}", "preds_test_sampling.npy"),
                allow_pickle=True
                ).item()

            prob_series = np.concatenate([arr[5:-5, :] for arr in prob_series_dict.values()], axis=0)
            true_series = np.concatenate(
                [Ys_test[item][5:-5, :] for item in Ys_test if item in prob_series_dict],
                axis=0
            )
            correlation = np.zeros(true_series.shape[1], dtype=np.float32)
            for p in range(len(correlation)):
                correlation[p] = pearsonr(true_series[:, p],
                    prob_series[:, p])[0]
                
            values = reduce_parcel_vector_to_areas(correlation, area_ids)
            for area_idx, val in enumerate(values):
                rows.append({
                    f"{ablation_type} window length (s)": window,
                    "r": float(val / noise_ceiling[0][area_idx]),
                    "network": network_names[area_idx] if network_names else area_idx,
                    "framework":  f"{framework}"
                })
            
            
            whole_brain_rows.append({
                f"{ablation_type} window length (s)":window,
                "r": float(compute_pearson_score_full(true_series, prob_series) / noise_ceilingwb[0][0]),
                "framework":  "AFM" if framework == "afm" else "SFM"
            })

    df_networks = pd.DataFrame(rows)
    df_whole_brain = pd.DataFrame(whole_brain_rows) 
    plt.figure(figsize=(6,3))
    sns.lineplot(
        data=df_whole_brain,
        x=f"{ablation_type} window length (s)",
        y="r",
        hue="framework",
        marker="o",
        palette=sns.color_palette("Set3")[3:5][::-1]
    )

    plt.legend(
        title='framework',
        bbox_to_anchor=(1.05, 1), 
        loc='upper left'
    )
    plt.show()

    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(12, 5),
        sharex=True, sharey=True
    )

    for i, fw in enumerate(frameworks):
        ax = axes[i]
        sns.lineplot(
            data=df_networks[df_networks["framework"] == fw],
            x=f"{ablation_type} window length (s)",
            y="r",
            hue="network",
            style="network",
            markers="o",
            dashes=False,
            palette=palette,
            ax = ax
            
        )
        ax.set_xlabel(f"{ablation_type} window length (s)")
        ax.set_ylabel("r*")
        ax.set_title("AFM" if fw == "afm" else "SFM", y=-0.25)

        ax.get_legend().remove() 
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(
        handles, labels,
        title="network",
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
    )


    plt.tight_layout()
    plt.show()


NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_TO_ID = {n: i for i, n in enumerate(NETWORK_ORDER)}

def load_schaefer7_area_ids_from_lut(lut_path, expected_parcels=None):
    """
    Parse LUT lines of the form:
      <idx> <R> <G> <B> <label>
    where label is like '7Networks_LH_Vis_1'.
    Returns:
      area_ids: (P,) int32 in 0..6
    """
    rows = []
    with open(lut_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split() 
            if len(parts) < 5:
                continue
            idx = int(parts[0])
            label = parts[-1]    
            rows.append((idx, label))

    if expected_parcels is None:
        P = max(idx for idx, _ in rows)
    else:
        P = expected_parcels

    area_ids = -1 * np.ones((P,), dtype=np.int32)

    for idx, label in rows:
        
        toks = label.split("_")
        net_token = next((t for t in toks if t in NET_TO_ID), None)
        if net_token is None:
            raise ValueError(f"Could not parse network from label: {label}")
        pos = idx - 1
        if pos < 0 or pos >= P:
            continue
        area_ids[pos] = NET_TO_ID[net_token]

    if (area_ids < 0).any():
        missing = np.where(area_ids < 0)[0]
        raise ValueError(f"{len(missing)} parcels missing network assignment; first few: {missing[:10]}")

    return area_ids

