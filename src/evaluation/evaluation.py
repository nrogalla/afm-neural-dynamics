from scipy.stats import pearsonr
import numpy as np

def compute_pearson_score_full(fmri_test, fmri_test_pred):
    """
    Computes Pearson correlation score averaged across parcels.
    """
    encoding_accuracy = np.zeros((fmri_test.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_test[:, p],
            fmri_test_pred[:, p])[0]
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)
    return mean_encoding_accuracy