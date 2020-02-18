import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import src.visualization.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click

hp.set_layout(15)

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))



def plot_corrcoeff(map,ax):

    c = ax.pcolormesh(map,vmin=-1.0,vmax=1.0)
    cbar = plt.colorbar(c)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('Correlation Coefficient',  labelpad=10, rotation=270)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_xticklabels(['', 'HRTF_C','HRTF_I','Learned MAP',''])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_yticklabels(['', 'HRTF_C','HRTF_I','Learned MAP',''])

    return ax



# Define whether figures should be saved
@click.command()
@click.option('--save_figs', default=False, help='Save the figures.')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
def main(save_figs=False, save_type='svg', model_name='hrtf_comparison', exp_name='single_participant'):

    logger = logging.getLogger(__name__)
    logger.info('Showing Correlation Coefficient Maps between HRTFs and differntly learned Maps')

    ### Set Parameters of Input Files ###
    azimuth = 12
    snr = 0.0
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
            [hrtfs_i,hrtfs_c,learned_map_mono,learned_map_mono_mean,learned_map_bin,learned_map_bin_mean] = pickle.load(f)

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(2,2,1)
        ax.set_title('Mono Map')
        tmp = np.concatenate((hrtfs_c, hrtfs_i, learned_map_mono))
        tmp = np.corrcoef(tmp)
        plot_corrcoeff(tmp,ax)

        ax = fig.add_subplot(2,2,2)
        ax.set_title('Mono-Mean Map')
        tmp = np.concatenate((hrtfs_c, hrtfs_i, learned_map_mono_mean))
        tmp = np.corrcoef(tmp)
        plot_corrcoeff(tmp,ax)


        ax = fig.add_subplot(2,2,3)
        ax.set_title('Bin Map')
        tmp = np.concatenate((hrtfs_c, hrtfs_i, learned_map_bin))
        tmp = np.corrcoef(tmp)
        plot_corrcoeff(tmp,ax)


        ax = fig.add_subplot(2,2,4)
        ax.set_title('Bin-Mean Map')
        tmp = np.concatenate((hrtfs_c, hrtfs_i, learned_map_bin_mean))
        tmp = np.corrcoef(tmp)
        plot_corrcoeff(tmp,ax)

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name
            logger.info('Saving Figures to : '+fig_save_path.as_posix())
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
