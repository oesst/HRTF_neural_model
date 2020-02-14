import matplotlib.pyplot as plt
import src.visualization.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))

# Define whether figures should be saved
@click.command()
@click.option('--save_figs', default=False, help='Save the figures.')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
def main(save_figs=False, save_type='svg',model_name='single_participant', exp_name = 'single_participant_different_azis'):

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    ### Set Parameters of Input Files ###
    azimuth = 13
    snr = 0.2
    freq_bands = 128
    participant_number = 9

    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, 25, 1)

    # filtering parameters
    normalization_type = 'sum_1'
    sigma_smoothing = 0
    sigma_gauss_norm = 1

    # use the mean subtracted map as the learned map
    mean_subtracted_map = True

    ear = 'ipsi'
    ######################################

    # create unique experiment name
    exp_name_str = exp_name + '_' + normalization_type + str(sigma_smoothing) + str(sigma_gauss_norm) + str(mean_subtracted_map) + '_' + str(time_window) + '_window_{0:03d}'.format(participant_number) + '_cipic_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm.npy'

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [x_mono, y_mono, x_mono_mean, y_mono_mean, x_bin,
                y_bin, x_bin_mean, y_bin_mean] = pickle.load(f)

        # define which elevations should be used
        print(y_mono.shape)
        x_mono = x_mono[:, elevations, :]
        y_mono = y_mono[:, elevations]
        x_mono_mean = x_mono_mean[:, elevations, :]
        y_mono_mean = y_mono_mean[:, elevations]
        x_bin = x_bin[:, elevations, :]
        y_bin = y_bin[:, elevations]
        x_bin_mean = x_bin_mean[:, elevations, :]
        y_bin_mean = y_bin_mean[:, elevations]

        fig = plt.figure(figsize=(20, 5))
        # plt.suptitle('Single Participant')
        # Monoaural Data (Ipsilateral), No Mean Subtracted
        ax = fig.add_subplot(1, 4, 1)
        hp.plot_localization_result(
            x_mono, y_mono, ax, SOUND_FILES, scale_values=True, linear_reg=True)
        ax.set_title('Monoaural')
        hp.set_axis(ax)
        ax.set_ylabel('Estimated Elevation [deg]')
        ax.set_xlabel('True Elevation [deg]')

        # Monoaural Data (Ipsilateral), Mean Subtracted
        ax = fig.add_subplot(1, 4, 2)
        hp.plot_localization_result(
            x_mono_mean, y_mono_mean, ax, SOUND_FILES, scale_values=True, linear_reg=True)
        ax.set_title('Mono - Mean')
        hp.set_axis(ax)
        ax.set_xlabel('True Elevation [deg]')

        # Binaural Data (Ipsilateral), No Mean Subtracted
        ax = fig.add_subplot(1, 4, 3)
        hp.plot_localization_result(
            x_bin, y_bin, ax, SOUND_FILES, scale_values=True, linear_reg=True)
        ax.set_title('Binaural')
        hp.set_axis(ax)
        ax.set_xlabel('True Elevation [deg]')

        # Binaural Data (Ipsilateral), Mean Subtracted
        ax = fig.add_subplot(1, 4, 4)
        hp.plot_localization_result(
            x_bin_mean, y_bin_mean, ax, SOUND_FILES, scale_values=True, linear_reg=True)
        ax.set_title('Bin - Mean')
        hp.set_axis(ax)
        ax.set_xlabel('True Elevation [deg]')

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig((fig_save_path / (exp_name+'_localization.' + save_type)).as_posix(), dpi=300)

        plt.show()
    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
