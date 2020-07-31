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
@click.option('--participant_number', default=9, help='CIPIC participant number. Default is 9')
@click.option('--snr', default=0.2, help='Signal to noise ration to use. Default is 0.2')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--sigma_smoothing', default=0.0, help='Sigma for smoothing kernel. 0 is off. Default is 0.')
@click.option('--sigma_gauss_norm', default=1.0, help='Sigma for gauss normalization. 0 is off. Default is 1.')
def main(save_figs=False, save_type='svg', model_name='single_participant', exp_name='single_participant_default', azimuth=12, participant_number=9, snr=0.2, freq_bands=24, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1):

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
        snr * 100), freq_bands, max_freq, participant_number, (azimuth - 12) * 10, normalize, len(elevations), ear])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [localization_results, q_ele_all, r_ipsi_all] = pickle.load(f)

        fig = plt.figure()

        axes = fig.subplots(1, 1, squeeze=False, sharex=True, sharey=True)

        ax1 = axes[0, 0]
        ax1.set_title('Neuron Results')
        for i_sound, sound in enumerate(SOUND_FILES):
            ax1.scatter(localization_results[i_sound, :, 0], localization_results[i_sound, :, 1])

        print(np.squeeze(localization_results[:, :, 0]).shape)
        print(np.squeeze(localization_results[:, :, 1]).shape)
        lr = hp_vis.LinearReg(np.squeeze(localization_results[:, :, 0]), np.squeeze(localization_results[:, :, 1]))
        x, y = lr.get_fitted_line()
        ax1.plot(x, y, linewidth=3, color='black')
        print('Neuron:')
        lr.print_coefficients()
        # ax1.set_ylim([0,25])
        # ax1.set_xlim([0,25])
        plt.show()

        plt.tight_layout()

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name / exp_name_str
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
