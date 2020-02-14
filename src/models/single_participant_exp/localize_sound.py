# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from src.data import generateData
from src.features import helpers as hp
from src.visualization import helpers as hpVis
import numpy as np
import pickle
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))

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


def process_inputs(psd_all_i, psd_all_c, normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1):
    # filter the data
    psd_mono_c = hp.filter_dataset(psd_all_c, normalization_type=normalization_type,
                                   sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)
    psd_mono_i = hp.filter_dataset(psd_all_i, normalization_type=normalization_type,
                                   sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)

    # integrate the signals and filter
    psd_binaural = hp.filter_dataset(
        psd_mono_i / psd_mono_c, normalization_type=normalization_type, sigma_smoothing=0, sigma_gauss_norm=0)

    # calculate different input sounds. should be 4 of them (mono,mono-mean,bin, bin-mean)
    psd_mono = psd_mono_i
    psd_mono_mean = psd_mono - \
        np.transpose(np.tile(np.mean(psd_mono, axis=1), [
                     psd_mono.shape[1], 1, 1]), [1, 0, 2])
    psd_binaural = psd_binaural
    psd_binaural_mean = psd_binaural - \
        np.transpose(np.tile(np.mean(psd_binaural, axis=1), [
                     psd_binaural.shape[1], 1, 1]), [1, 0, 2])

    return psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    ### Set Parameters of Input Files ###
    azimuth = 13
    snr = 0.2
    freq_bands = 128
    participant_number = 9

    normalize = False
    time_window = 0.1  # time window in sec

    # filtering parameters
    normalization_type = 'sum_1'
    sigma_smoothing = 0
    sigma_gauss_norm = 1

    # use the mean subtracted map as the learned map
    mean_subtracted_map = True

    ######################################

    # create unique experiment name
    exp_name = 'single_participant'
    exp_name_str = exp_name + '_' + normalization_type + str(sigma_smoothing) + str(sigma_gauss_norm) + str(mean_subtracted_map) + '_' + str(time_window) + '_window_{0:03d}'.format(participant_number) + '_cipic_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_' + str((azimuth - 13) * 10) + '_azi_' + str(normalize) + '_norm.npy'

    exp_path = ROOT / 'models' / exp_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [x_mono, y_mono, x_mono_mean, y_mono_mean, x_bin,
                y_bin, x_bin_mean, y_bin_mean] = pickle.load(f)
    else:
        # create Path
        exp_path.mkdir(parents=True, exist_ok=True)
        # create or read the data
        psd_all_c, psd_all_i = generateData.create_data(
            freq_bands, participant_number, snr, normalize, azimuth, time_window)

        # filter data and integrate it
        psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = process_inputs(
            psd_all_i, psd_all_c, normalization_type, sigma_smoothing, sigma_gauss_norm)

        # create map from defined processed data
        learned_map = create_map(psd_binaural, mean_subtracted_map)

        # localize the sounds and save the results
        x_mono, y_mono = hp.localize_sound(psd_mono, learned_map)

        # localize the sounds and save the results
        x_mono_mean, y_mono_mean = hp.localize_sound(
            psd_mono_mean, learned_map)

        # localize the sounds and save the results
        x_bin, y_bin = hp.localize_sound(psd_binaural, learned_map)

        # localize the sounds and save the results
        x_bin_mean, y_bin_mean = hp.localize_sound(
            psd_binaural_mean, learned_map)

        with open(exp_file.as_posix(), 'wb') as f:
            logger.info('Creating model file')
            pickle.dump([x_mono, y_mono, x_mono_mean, y_mono_mean,
                         x_bin, y_bin, x_bin_mean, y_bin_mean], f)

    fig = plt.figure(figsize=(20, 5))
    # plt.suptitle('Single Participant')
    # Monoaural Data (Ipsilateral), No Mean Subtracted
    ax = fig.add_subplot(1, 4, 1)
    hpVis.plot_localization_result(x_mono, y_mono, ax, SOUND_FILES, scale_values=True, linear_reg=True)
    ax.set_title('Monoaural')
    hp.set_axis(ax)
    ax.set_ylabel('Estimated Elevation [deg]')
    plt.show()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
