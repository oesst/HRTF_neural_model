# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from os import listdir
from os.path import isfile, join

import numpy as np
import soundfile as sf
from scipy import io
import scipy.signal as sp
from src.features import gtgram

ROOT = Path(__file__).resolve().parents[2]
# set the path to the sound files
# create a list of the sound files


def create_data(freq_bands=24, participant_number=19, snr=0.2, normalize=False, azimuth=13, time_window=0.1):

    str_r = 'data/processed/binaural_right_0_gammatone_' + str(time_window) + '_window_{0:03d}'.format(participant_number) + '_cipic_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm_flat_spectrum.npy'
    str_l = 'data/processed/binaural_left_0_gammatone_' + str(time_window) + '_window_{0:03d}'.format(participant_number) + '_cipic_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm_flat_spectrum.npy'

    path_data_r = ROOT / str_r
    path_data_l = ROOT / str_l

    # Default gammatone-based spectrogram parameters
    twin = time_window
    thop = twin / 2
    fmin = 100
    fs = 44100

    # check if we can load the data from a file
    if path_data_r.is_file() and path_data_l.is_file():
        print('Data set found. Loading from file : ' + str_r)
        return np.load(path_data_r), np.load(path_data_l)
    else:
        print('Creating HRTFs : ' + str_l)
        # read the HRIR data
        hrtf_path = (
            ROOT / 'data/raw/hrtfs/hrir_{0:03d}.mat'.format(participant_number)).resolve()
        hrir_mat = io.loadmat(hrtf_path.as_posix())

        # get the data for the left ear
        hrir_l = hrir_mat['hrir_l']
        # get the data for the right ear
        hrir_r = hrir_mat['hrir_r']
        # use always all elevations -> 50
        psd_all_i = np.zeros((1,50, freq_bands))
        psd_all_c = np.zeros((1,50, freq_bands))
        # temporal_means = np.zeros((hrir_elevs.shape[0],87))
        for i_elevs in range(psd_all_i.shape[1]):
            # read the hrir for a specific location
            hrir_elevs = np.squeeze(hrir_l[azimuth, i_elevs, :])
            # use a flat spectrum
            signal = np.ones(fs)

            # add noise to the signal
            signal_elevs = (1 - snr) * sp.lfilter(hrir_elevs, 1, signal) + \
                snr * (signal + np.random.random(signal.shape[0]) * snr)

            ###### TAKE THE ENTIRE SIGNAL #######
            #         window_means = get_spectrum(signal_elevs,nperseg=welch_nperseg)
            #####################################
            # read the hrir for a specific location
            hrir_elevs = np.squeeze(hrir_r[azimuth, i_elevs, :])

            # add noise to the signal
            signal_elevs_c = (1 - snr) * sp.lfilter(hrir_elevs, 1, signal) + \
                snr * (signal + np.random.random(signal.shape[0]) * snr)

            ###### Apply Gammatone Filter Bank ##############
            y = gtgram.gtgram(signal_elevs, fs, twin,
                              thop, freq_bands, fmin)
            y = (20 * np.log10(y + 1))
            window_means = np.mean(y, axis=1)
            psd_all_i[0,i_elevs, :] = window_means

            y = gtgram.gtgram(signal_elevs_c, fs,
                              twin, thop, freq_bands, fmin)
            y = (20 * np.log10(y + 1))
            window_means = np.mean(y, axis=1)
            psd_all_c[0,i_elevs, :] = window_means
            #################################################

        np.save(path_data_r.absolute(), psd_all_c)
        np.save(path_data_l.absolute(), psd_all_i)

        return psd_all_c, psd_all_i


def main():
    """ This script creates data of pure HRTFs.
    That is, a flat signal (np.ones) is filtered with a HRTF and then gammatone transformed to a frequency spectrum.
    This frequency spectrum resembles the HRTF of a participant.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #### Set parameters ####
    ########################
    normalize = False  # paramter is not considered

    time_window = 0.1  # time window for spectrogram in sec

    # Parameter to test
    # snrs = np.arange(0, 1.1, 0.1)
    snrs = np.array([0])
    # freq_bandss = np.array([32, 64, 128])
    freq_bandss = np.array([128])
    # azimuths = np.arange(0, 25, 1)
    azimuths = np.array([12])
    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
                                    12, 15, 17, 18, 19, 20, 21, 27, 28, 33, 40])

    # walk over all parameter combinations
    for _, participant_number in enumerate(participant_numbers):
        for _, snr in enumerate(snrs):
            for _, freq_bands in enumerate(freq_bandss):
                for _, azimuth in enumerate(azimuths):
                    psd_all_c, psd_all_i = create_data(
                        freq_bands, participant_number, snr, normalize, azimuth, time_window)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
