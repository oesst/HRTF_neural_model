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


def process_inputs(psd_all_i, psd_all_c, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1):
    # filter the data
    psd_mono_c = hp.filter_dataset(psd_all_c, normalization_type=normalization_type,
                                   sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)
    psd_mono_i = hp.filter_dataset(psd_all_i, normalization_type=normalization_type,
                                   sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)

    # integrate the signals and filter
    if ear.find('contra') >= 0:
        psd_binaural = hp.filter_dataset(
            psd_mono_c / psd_mono_i, normalization_type=normalization_type, sigma_smoothing=0, sigma_gauss_norm=0)
    else:
        psd_binaural = hp.filter_dataset(
            psd_mono_i / psd_mono_c, normalization_type=normalization_type, sigma_smoothing=0, sigma_gauss_norm=0)

    # calculate different input sounds. should be 4 of them (mono,mono-mean,bin, bin-mean)
    if ear.find('contra') >= 0:
        psd_mono = psd_mono_c
    else:
        psd_mono = psd_mono_i

    psd_mono_mean = psd_mono - \
        np.transpose(np.tile(np.mean(psd_mono, axis=1), [
                     psd_mono.shape[1], 1, 1]), [1, 0, 2])
    psd_binaural = psd_binaural
    psd_binaural_mean = psd_binaural - \
        np.transpose(np.tile(np.mean(psd_binaural, axis=1), [
                     psd_binaural.shape[1], 1, 1]), [1, 0, 2])

    return psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean


def main(model_name='single_participant', exp_name='single_participant_different_azis'):
    """ Localizes sounds at azimuth 'azimuth' with a learned map at azimuth 0.
    """
    logger = logging.getLogger(__name__)
    logger.info('Localizing sounds for a single participant at different azimuths')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
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

    # choose which ear to use 'contra' or 'ipsi'
    ear = 'ipsi'

    ########################################################################
    ########################################################################

    exp_name_str = exp_name + '_' + normalization_type + str(sigma_smoothing) + str(sigma_gauss_norm) + str(mean_subtracted_map) + '_' + str(time_window) + '_window_{0:03d}'.format(participant_number) + '_cipic_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm.npy'

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with exp_file.open('rb') as f:
            logger.info('Reading model data from file')
            [x_mono, y_mono, x_mono_mean, y_mono_mean, x_bin,
                y_bin, x_bin_mean, y_bin_mean] = pickle.load(f)
    else:
        # create Path
        exp_path.mkdir(parents=True, exist_ok=True)
        # create or read the data
        psd_all_c, psd_all_i = generateData.create_data(
            freq_bands, participant_number, snr, normalize, 12, time_window)

        ####### Map Learning #######
        # filter data and integrate it for map learning
        psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = process_inputs(
            psd_all_i, psd_all_c, ear, normalization_type, sigma_smoothing, sigma_gauss_norm)

        # create map from defined processed data
        learned_map = hp.create_map(psd_binaural, mean_subtracted_map)

        ####### Input Processing #######
        # process data for actual input
        psd_all_c, psd_all_i = generateData.create_data(
            freq_bands, participant_number, snr, normalize, azimuth, time_window)

        # filter data and integrate it
        psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = process_inputs(
            psd_all_i, psd_all_c, ear, normalization_type, sigma_smoothing, sigma_gauss_norm)


        ####### Localization #######
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

        with exp_file.open('wb') as f:
            logger.info('Creating model file')
            pickle.dump([x_mono, y_mono, x_mono_mean, y_mono_mean,
                         x_bin, y_bin, x_bin_mean, y_bin_mean], f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
