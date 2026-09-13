import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy import stats


def add_nans_uniform_everywhere(data, p, rng):
    missing_matrix = rng.choice([np.nan, 1.0], size=data.shape, p=[p, 1 - p])
    return data * missing_matrix


def add_nans_uniform_specific_samples(data, affected_samples, p, rng):
    n_samples, n_features = data.shape
    missing_matrix = np.zeros(data.shape, dtype=bool)
    for i in affected_samples:
        missing_matrix[i] = rng.choice([True, False], size=(n_features), p=[p, 1 - p])
    d_miss = data.copy()
    d_miss[missing_matrix] = np.nan
    return d_miss


def add_nans_uniform_partial(data, p, q, rng):
    n_samples, n_features = data.shape
    affected_samples = rng.choice(n_samples, size=(int(n_samples * q)))
    return add_nans_uniform_specific_samples(data, affected_samples, p, rng)


def coph_corr(computed_linkage, reference_pdist):
    return stats.pearsonr(reference_pdist, cophenet(computed_linkage)).statistic


def add_nans_uniform_only_lowerhalf(data, p, rng):
    n_samples, n_features = data.shape
    m = n_samples * n_features // 2
    threshold = np.partition(data.flatten(), m - 1)[m - 1]
    mask = data <= threshold
    d_miss = d.copy()
    d_miss[mask] = np.nan


def add_nans_uniform_only_lowerhalf(data, p, rng):
    # Mask lower half of all values
    n_samples, n_features = data.shape
    m = n_samples * n_features // 2
    threshold = np.partition(data.flatten(), m-1)[m-1]
    mask = data <= threshold

    # Apply p missingness randomly on all masked values
    q = p * np.sum(mask) / (n_samples * n_features) # TODO: Sanity Check before run
    d_miss = data.copy()
    missing_matrix = rng.choice([np.nan, 1.0], size=data.shape, p=[q, 1-q])
    d_miss[mask] *= missing_matrix[mask]
    return d_miss
