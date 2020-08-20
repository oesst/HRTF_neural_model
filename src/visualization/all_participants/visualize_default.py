import matplotlib.pyplot as plt
import src.features.helpers_vis as hp_vis
import src.features.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click


hp_vis.set_layout(15)


ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))

# Define whether figures should be saved
@click.command()
@click.option('--save_figs', type=click.BOOL, default=False, help='Save figures')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
@click.option('--model_name', default='single_participant', help='Defines the model name.')
@click.option('--exp_name', default='single_participant_default', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--snr', default=0.2, help='Signal to noise ration to use. Default is 0.2')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--sigma_smoothing', default=0.0, help='Sigma for smoothing kernel. 0 is off. Default is 0.')
@click.option('--sigma_gauss_norm', default=1.0, help='Sigma for gauss normalization. 0 is off. Default is 1.')
def main(save_figs=False, save_type='svg', model_name='localize_all_participant', exp_name='all_participants_default', azimuth=12, snr=0.2, freq_bands=24, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1):

    logger = logging.getLogger(__name__)
    logger.info('Showing localization results for a single participant')

    # make sure save type is given
    if not save_type or len(save_type) == 0:
        save_type = 'svg'

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, elevations, 1)

    ########################################################################
    ########################################################################

    exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map, time_window, int(
        snr * 100), freq_bands, max_freq, (azimuth - 12) * 10, normalize, len(elevations), ear])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [localization_results_binaural, localization_results_monaural, q_ele_all, r_ipsi_all] = pickle.load(f)

        fig = plt.figure(figsize=(20, 5))
        # plt.suptitle('Single Participant')
        # Monoaural Data (Ipsilateral), No Mean Subtracted
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        # get the right values
        x_mono = localization_results_monaural[:, :, :, 0]
        y_mono = localization_results_monaural[:, :, :, 1]
        x_mono_mean = localization_results_monaural[:, :, :, 0]
        y_mono_mean = localization_results_monaural[:, :, :, 2]

        x_bin = localization_results_binaural[:, :, :, 0]
        y_bin = localization_results_binaural[:, :, :, 1]
        x_bin_mean = localization_results_binaural[:, :, :, 0]
        y_bin_mean = localization_results_binaural[:, :, :, 2]

        # plot regression line for each participant
        for i_par in range(x_mono.shape[0]):

            hp_vis.plot_localization_result(x_mono[i_par], y_mono[i_par], ax1, SOUND_FILES, scale_values=True, linear_reg=True, scatter_data=False)
            ax1.set_title('Monoaural')
            hp_vis.set_axis(ax1, len(elevations))
            ax1.set_ylabel('Estimated Elevation [deg]')
            ax1.set_xlabel('True Elevation [deg]')

            # Monoaural Data (Ipsilateral), Mean Subtracted
            hp_vis.plot_localization_result(x_mono_mean[i_par], y_mono_mean[i_par], ax2, SOUND_FILES,
                                            scale_values=True, linear_reg=True, scatter_data=False)
            ax2.set_title('Mono - Prior')
            hp_vis.set_axis(ax2, len(elevations))
            ax2.set_xlabel('True Elevation [deg]')

            # Binaural Data (Ipsilateral), No Mean Subtracted

            hp_vis.plot_localization_result(x_bin[i_par], y_bin[i_par], ax3, SOUND_FILES, scale_values=True, linear_reg=True, scatter_data=False)
            ax3.set_title('Binaural')
            hp_vis.set_axis(ax3, len(elevations))
            ax3.set_xlabel('True Elevation [deg]')

            # Binaural Data (Ipsilateral), Mean Subtracted

            hp_vis.plot_localization_result(x_bin_mean[i_par], y_bin_mean[i_par], ax4, SOUND_FILES,
                                            scale_values=True, linear_reg=True, scatter_data=False)
            ax4.set_title('Bin - Prior')
            hp_vis.set_axis(ax4, len(elevations))
            ax4.set_xlabel('True Elevation [deg]')

        # plot a common regression line
        x_mono_ = np.reshape(x_mono, (x_mono.shape[0] * x_mono.shape[1], x_mono.shape[2]))
        y_mono_ = np.reshape(y_mono, (y_mono.shape[0] * y_mono.shape[1], y_mono.shape[2]))

        x_mono_mean_ = np.reshape(x_mono_mean, (x_mono_mean.shape[0] * x_mono_mean.shape[1], x_mono_mean.shape[2]))
        y_mono_mean_ = np.reshape(y_mono_mean, (y_mono_mean.shape[0] * y_mono_mean.shape[1], y_mono_mean.shape[2]))

        x_bin_ = np.reshape(x_bin, (x_bin.shape[0] * x_bin.shape[1], x_bin.shape[2]))
        y_bin_ = np.reshape(y_bin, (y_bin.shape[0] * y_bin.shape[1], y_bin.shape[2]))

        x_bin_mean_ = np.reshape(x_bin_mean, (x_bin_mean.shape[0] * x_bin_mean.shape[1], x_bin_mean.shape[2]))
        y_bin_mean_ = np.reshape(y_bin_mean, (y_bin_mean.shape[0] * y_bin_mean.shape[1], y_bin_mean.shape[2]))

        hp_vis.plot_localization_result(x_mono_, y_mono_, ax1, SOUND_FILES, scale_values=True, linear_reg=True,
                                        disp_values=True, scatter_data=False, reg_color="black")
        hp_vis.plot_localization_result(x_mono_mean_, y_mono_mean_, ax2, SOUND_FILES, scale_values=True,
                                        linear_reg=True, disp_values=True, scatter_data=False, reg_color="black")
        hp_vis.plot_localization_result(x_bin_, y_bin_, ax3, SOUND_FILES, scale_values=True, linear_reg=True,
                                        disp_values=True, scatter_data=False, reg_color="black")
        hp_vis.plot_localization_result(x_bin_mean_, y_bin_mean_, ax4, SOUND_FILES, scale_values=True,
                                        linear_reg=True, disp_values=True, scatter_data=False, reg_color="black")

        plt.tight_layout()

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / exp_name_str / model_name
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig((fig_save_path / (model_name + '_' + exp_name + '_localization.' + save_type)).as_posix(), dpi=300)

        else:
            plt.show()
    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
