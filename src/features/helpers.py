import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch
from itertools import islice
import matplotlib as mpl
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import seaborn as sns



def localize_sound(psd_all, data_to_compare, metric='correlation'):

    x_test = np.zeros((psd_all.shape[0], psd_all.shape[1], 2))
    y_test = np.zeros((psd_all.shape[0], psd_all.shape[1]))

    # walk over all samples
    for i in range(psd_all.shape[0]):
        for ii in range(psd_all.shape[1]):
            data_point = np.squeeze(psd_all[i, ii, :])
            data_point = data_point[np.newaxis]

            # search for smallest distance between white noise data points and the randomly choosen sound sample
            dists = distance.cdist(data_to_compare, data_point, metric=metric)
            minimal_dist_ind = np.argmin(dists)
            y_test[i, ii] = minimal_dist_ind

            # save the data points
            x_test[i, ii, 0] = i
            x_test[i, ii, 1] = ii

    return x_test, y_test


def create_map(psd, mean_subtracted_map=True):

    # set the map
    if mean_subtracted_map:
        learned_map = np.mean(psd, axis=0)
        mean_learned_map = np.mean(learned_map, axis=0)
        learned_map = learned_map - mean_learned_map
    else:
        learned_map = np.mean(psd, axis=0)
        learned_map = learned_map

    return learned_map



def filter_dataset(dataset, normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=0):
    # applies filter and normalization on dataset. normalization_type can be one of the following: 'sum_1','l2','l1'
    ds = np.copy(dataset)
    # first do a smoothing
    if sigma_smoothing > 0:
        ds = gaussian_filter1d(ds, sigma=sigma_smoothing, mode='nearest', axis=2)

    # now normalize the data
    if normalization_type.find('sum_1') >= 0:
        # print('Sum 1 Normalization')
        ds = (ds / np.transpose(np.tile(np.sum(ds, axis=2), [ds.shape[2], 1, 1]), [1, 2, 0]))
    elif normalization_type.find('l1') >= 0:
        # print('L1 Normalization')
        ds = (ds - np.transpose(np.tile(np.min(ds, axis=2), [ds.shape[2], 1, 1]), [1, 2, 0]))
        ds = (ds / np.transpose(np.tile(np.max(ds, axis=2), [ds.shape[2], 1, 1]), [1, 2, 0]))
    elif normalization_type.find('l2') >= 0:
        # print('L2 Normalization')
        ds = (ds / np.transpose(np.tile(np.sqrt(np.sum(ds ** 2, axis=2)), [ds.shape[2], 1, 1]), [1, 2, 0]))
    else:
        # print('No Normalization')
        ds = ds

        # now divide the data by a gaussian convolved version of it (filter it)
    if sigma_gauss_norm > 0:
        ds = ds / (gaussian_filter1d(ds, sigma=sigma_gauss_norm, mode='nearest', axis=2))

    return ds
