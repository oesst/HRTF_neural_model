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

import librosa

ROOT = Path(__file__).resolve().parents[2]
# set the path to the sound files
# create a list of the sound files
# set the path to the sound files
SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))
# Define up to which frequency the data should be generated



def create_data(freq_bands=24, participant_number=19, snr=0.2, normalize=False, azimuth=13, time_window=0.1, max_freq=18000, clean=False):

    dir_name = ROOT / ('data/processed_' + str(max_freq) + 'Hz/')

    str_r = 'binaural_right_0_gammatone_' + str(time_window) + '_window_{0:03d}'.format(participant_number) + '_cipic_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm_flat_spectrum_stft.npy'
    str_l = 'binaural_left_0_gammatone_' + str(time_window) + '_window_{0:03d}'.format(participant_number) + '_cipic_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm_flat_spectrum_stft.npy'

    path_data_r = dir_name / str_r
    path_data_l = dir_name / str_l

    # Default gammatone-based spectrogram parameters
    twin = time_window
    thop = twin / 2
    fmin = 20
    fmax = max_freq
    fs = 44100

    # check if we can load the data from a file
    if not clean and path_data_r.is_file() and path_data_l.is_file():
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
        psd_all_i = np.zeros((1, 50, int(freq_bands/2+1)))
        psd_all_c = np.zeros((1, 50, int(freq_bands/2+1)))
        # temporal_means = np.zeros((hrir_elevs.shape[0],87))
        for i_elevs in range(psd_all_i.shape[1]):
            # read the hrir for a specific location
            hrir_elevs = np.squeeze(hrir_l[azimuth, i_elevs, :])
            # use a flat spectrum
            signal = np.ones(fs)
            # print(SOUND_FILES[18].as_posix())
            # signal = sf.read(SOUND_FILES[18].as_posix())[0]
            # if we make twin depending on the length of the sound we do not have to take the mean value

            # add noise to the signal
            signal_elevs = (1 - snr) * sp.lfilter(hrir_elevs, 1, signal) + \
                snr * (signal + np.random.random(signal.shape[0]) * snr)

            # read the hrir for a specific location
            hrir_elevs = np.squeeze(hrir_r[azimuth, i_elevs, :])

            # add noise to the signal
            signal_elevs_c = (1 - snr) * sp.lfilter(hrir_elevs, 1, signal) + \
                snr * (signal + np.random.random(signal.shape[0]) * snr)

            y = np.abs(librosa.stft(signal_elevs,n_fft=freq_bands))
            window_means = np.mean(y, axis=1)
            psd_all_i[0, i_elevs, :] = np.log10(window_means)*20


            y = np.abs(librosa.stft(signal_elevs_c,n_fft=freq_bands))
            window_means = np.mean(y, axis=1)
            psd_all_c[0, i_elevs, :] = np.log10(window_means)*20

        freqs = librosa.fft_frequencies(sr=fs, n_fft=freq_bands)


        # cut off frequencies at the end and beginning 100
        indis = np.logical_and(100 <= freqs, freqs < max_freq)
        # f_original = freqs[indis]
        psd_all_i = psd_all_i[:,:,indis]
        psd_all_c = psd_all_c[:,:,indis]

        # indis = f_original >= 100
        # f_original = f_original[indis]
        # psd_original = psd_original[indis]


        # create directory
        dir_name.mkdir(exist_ok=True)
        dir_name.mkdir(exist_ok=True)
        # save data
        np.save(path_data_r.absolute(), psd_all_c)
        np.save(path_data_l.absolute(), psd_all_i)

        return psd_all_c, psd_all_i, freqs


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
