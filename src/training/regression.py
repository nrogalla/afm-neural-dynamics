import os
import numpy as np
import h5py

from sklearn.linear_model import LinearRegression

import utils.utils_combi as utils
from evaluation.evaluation import compute_pearson_score_full
from data.dataloaders import load_subject_data
import utils.runconfigs as rc
import data.dataloaders as dl
import utils.utils_combi as utils
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

def train_encoding(features_train, fmri_train):
    """
    Train a linear-regression-based encoding model to predict fMRI responses
    using movie features.

    Parameters
    ----------
    features_train : float
        Stimulus features for the training movies.
    fmri_train : float
        fMRI responses for the training movies.

    Returns
    -------
    model : object
        Trained regression model.

    """

    model = LinearRegression().fit(features_train, fmri_train)

    return model

def get_data_single_subject_regression(runconfig, modality, area):
    features = utils.load_stimulus_features(runconfig.root_dir,modality)
    fmri = utils.load_fmri(runconfig.root_dir, runconfig.subject_id)
    fmri =  dl.apply_voxel_selection(fmri, area)
    fmri, features = dl.sort_data(fmri, features)

    def get_data_regression(movies):
        aligned_features, aligned_fmri = utils.align_features_and_fmri_samples(features, fmri, runconfig.hrf_delay, movies, 0, runconfig.segment_length_steps)

        aligned_features_hrf = []
        for features_split in aligned_features:
            aligned_features_hrf.append(features_split[:-2])
        features_reg = np.concatenate(aligned_features_hrf, axis=0)
        fmri_reg = np.concatenate(aligned_fmri, axis=0)
        return features_reg, fmri_reg

    features_train, fmri_train = get_data_regression(runconfig.movies_train)
    del features, fmri
    return features_train, fmri_train

def get_data_single_subject_regression_pred(runconfig, modality, area):
    features = utils.load_stimulus_features(runconfig.root_dir,modality)
    fmri = utils.load_fmri(runconfig.root_dir, runconfig.subject_id)
    fmri =  dl.apply_voxel_selection(fmri, area)
    fmri, features = dl.sort_data(fmri, features)

    def get_data_regression(movies):
        aligned_features, _ = utils.align_features_and_fmri_samples(features, fmri, runconfig.hrf_delay, movies, 0, runconfig.segment_length_steps)
        return aligned_features
    aligned_features_01 = get_data_regression(["friends-s01"])
    
    aligned_features_test = get_data_regression(runconfig.movies_test)
    return fmri, aligned_features_01, aligned_features_test

def get_score(pred_dict,fmri):
        preds = []
        y_true = []
        for i, (epi, val) in enumerate(pred_dict.items()): 
            preds.append(val[5:-5, :])
            y_true.append(fmri[epi][5:-5, :])

        preds = np.concatenate(preds, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        correlation = compute_pearson_score_full(y_true, preds)
        return correlation

runconfig = rc.BaseRunConfigs(None, None)

modality = "all"
area = None

subjects = [1, 2, 3, 5]

all_features_train = []
all_fmri_train = []
all_features_01 = []
all_fmri_01 = []
all_features_test = []
all_fmri_test = []

test_correlation_per_subject = []
train_correlation_per_subject = []
train_individual_models = True

preds01 = {}
preds_test = {}

if train_individual_models == False:
    for subject in subjects:
        runconfig.subject_id = subject
        features_train, fmri_train = get_data_single_subject_regression(runconfig, modality, area)

        all_features_train.append(features_train)
        all_fmri_train.append(fmri_train)

    features_train = np.concatenate(all_features_train, axis = 0)
    fmri_train = np.concatenate(all_fmri_train, axis = 0)

    # if not train_individual_models:
    model = train_encoding(features_train, fmri_train)


for sub in subjects:
    outfolder = f"outputs_regression/s{sub}"
    os.makedirs(outfolder, exist_ok=True)
    runconfig.subject_id = sub
    fmri, aligned_features_01, aligned_features_test = get_data_single_subject_regression_pred(runconfig, modality, area)
    fmri_01 = {k: v for k, v in fmri.items() if "s01e" in k}
    fmri_test = {k: v for k, v in fmri.items() if "s06e" in k}
    if train_individual_models:
        runconfig.subject_id = sub
        features_train, fmri_train = get_data_single_subject_regression(runconfig, modality, area)

        model = train_encoding(features_train, fmri_train)

    # Trainset season 1
    preds01[sub] = {}
    preds_test[sub] = {}
    for ((epi, _), feat_movie) in zip(fmri_01.items(), aligned_features_01):
        fmri_pred = model.predict(feat_movie[:-runconfig.hrf_delay]).astype(np.float32)
        preds01[sub][epi] = fmri_pred
    # Testset
    for ((epi, _), feat_movie) in zip(fmri_test.items(), aligned_features_test):
        fmri_pred = model.predict(feat_movie[:-runconfig.hrf_delay]).astype(np.float32)
        preds_test[sub][epi] = fmri_pred
    np.save(f"{outfolder}/outputs_regr_train",preds01[sub] , allow_pickle=True)
    np.save(f"{outfolder}/outputs_regr_test",preds_test[sub], allow_pickle=True)
    
    train_correlation = get_score(preds01[sub], fmri_01)
    train_correlation_per_subject.append(train_correlation)
    test_correlation = get_score(preds_test[sub], fmri_test)
    test_correlation_per_subject.append(test_correlation)
    logging.info(preds_test[sub][epi].shape)
    logging.info(f"correlation train: {train_correlation}")
    logging.info(f"correlation test: {test_correlation}")
    

logging.info(f"Mean Train Correlation across subjects: {np.mean(np.array(train_correlation_per_subject))}")
logging.info(f"Mean Test Correlation across subjects: {np.mean(np.array(test_correlation_per_subject))}")