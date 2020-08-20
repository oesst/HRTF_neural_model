# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from src.data import generateData
from src.features import helpers as hp
from src.features import helpers_vis as hpVis
from src.features import network
import numpy as np
import pickle
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))


# Define whether figures should be saved
@click.command()
@click.option('--model_name', default='single_participant', help='Defines the model name.')
@click.option('--exp_name', default='single_participant_default', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--snr', default=0.0, help='Signal to noise ration to use. Default is 0.2')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--sigma_smoothing', default=0.0, help='Sigma for smoothing kernel. 0 is off. Default is 0.')
@click.option('--sigma_gauss_norm', default=1.0, help='Sigma for gauss normalization. 0 is off. Default is 1.')
@click.option('--sigma_gauss_norm', default=1.0, help='Sigma for gauss normalization. 0 is off. Default is 1.')
@click.option('--steady_state', is_flag=True)
@click.option('--clean', is_flag=True)
def main(model_name='train_network_single_participant', exp_name='single_participant_default', azimuth=12,  snr=0.0, freq_bands=24, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1, clean=False, steady_state=False):
    """ This script takes the filtered data and tries to localize sounds with a learned map
        for a single participant.
    """
    logger = logging.getLogger(__name__)
    logger.info('Training networks for all participants')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    normalize = False
    time_window = 0.1  # time window in sec

    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
                                    12, 15, 17, 18, 19, 20,
                                    21, 27, 28, 33, 40, 44,
                                    48, 50, 51, 58, 59, 60,
                                    61, 65, 119, 124, 126,
                                    127, 131, 133, 134, 135,
                                    137, 147, 148, 152, 153,
                                    154, 155, 156, 158, 162,
                                    163, 165])

    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
                                    12, 15, 17, 18, 19])

    elevations = np.arange(0, elevations, 1)
    ########################################################################
    ########################################################################

    for i_par, participant_number in enumerate(participant_numbers):

        # create unique experiment name
        exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map, time_window, int(
            snr * 100), freq_bands, max_freq, participant_number, (azimuth - 12) * 10, normalize, len(elevations), ear])

        exp_path = ROOT / 'models' / model_name
        exp_file = exp_path / (exp_name_str + '_weights')

        # check if model results exist already and load
        if not clean and exp_path.exists() and exp_file.is_file():
            # try to load the model files
            with exp_file.open('rb') as f:
                logger.info('Reading model data from file')
                [w, w_sounds_i, w_sounds_c] = pickle.load(f)
        else:

            # create Path
            exp_path.mkdir(parents=True, exist_ok=True)
            # create or read the data
            psd_all_c, psd_all_i = generateData.create_data(
                freq_bands, participant_number, snr, normalize, azimuth, time_window, max_freq=max_freq)

            # Take only given elevations
            input_c = psd_all_c[:, elevations, :]
            input_i = psd_all_i[:, elevations, :]

            # normalize inputs over frequencies
            input_c = input_c / input_c.sum(2)[:, :, np.newaxis]
            input_i = input_i / input_i.sum(2)[:, :, np.newaxis]

            # initialize network. if steady_state is True run do not use euler but calculate the response immediatley
            net = network.Network(steady_state=steady_state)

            # if we use the steady state response to learn, we need more trials
            if steady_state:
                trials = 1500 * 20
            else:
                trials = 25

            print(participant_number)

            for ele in range(trials):
                # for i_ele, ele in enumerate(elevations):
                ele = np.random.randint(0, len(elevations))
                sound = np.random.randint(0, len(SOUND_FILES))
                # sound = 1
                # ele = 1
                in_i = input_i[sound, ele]
                in_c = input_c[sound, ele]
                net.run(in_i, in_c, ele, sound, train=True, prior_info=True)

                # logger.info('Sound No: ' + str(sound + 1) + ' of ' + str(len(SOUND_FILES)) +
                #             '.  -> Elevation : ' + str(ele + 1) + ' of ' + str(len(elevations)))

            with exp_file.open('wb') as f:
                logger.info('Creating model file')
                pickle.dump([net.w, net.w_sounds_i, net.w_sounds_c], f)

        # # print(net.dt)
        # fig = plt.figure(figsize=(10, 5))
        # # # plt.suptitle('Single Participant')
        # # # Monoaural Data (Ipsilateral), No Mean Subtracted
        # ax = fig.add_subplot(2, 2, 1)
        # c = ax.pcolormesh(net.w)
        # plt.colorbar(c)
        #
        # ax = fig.add_subplot(2, 2, 2)
        # c = ax.pcolormesh(net.w_sounds_i[1])
        # plt.colorbar(c)
        #
        # ax = fig.add_subplot(2, 2, 3)
        # c = ax.pcolormesh(net.w_sounds_i[7])
        # plt.colorbar(c)
        #
        # ax = fig.add_subplot(2, 2, 4)
        # c = ax.pcolormesh(net.w_sounds_i[8])
        # plt.colorbar(c)
        #
        # plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
