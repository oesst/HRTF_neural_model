import matplotlib.pyplot as plt
import src.visualization.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click
import seaborn as sns

hp.set_layout(15)

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))


def get_regression_values(x, y):
    x, y = hp.scale_v(x, y)
    x = np.reshape(x, (x.shape[0] * x.shape[1], 2))
    y = np.reshape(y, (y.shape[0] * y.shape[1]))

    lr_model = hp.LinearReg(x, y)
    c_m, c_b = lr_model.get_coefficients()
    score = lr_model.get_score()

    return c_m, c_b, score


# Define whether figures should be saved
@click.command()
@click.option('--save_figs', default=False, help='Save the figures.')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
def main(save_figs=False, save_type='svg', model_name='all_participants', exp_name='localization_default'):

    logger = logging.getLogger(__name__)
    logger.info('Showing localization results for all participants')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    azimuth = 12
    snr = 0.2
    freq_bands = 128
    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
                                    12, 15, 17, 18, 19, 20, 21, 27, 28, 33, 40])

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
    elevations = np.arange(0, 25, 1)
    ########################################################################
    ########################################################################

    # create unique experiment name
    exp_name_str = exp_name + '_' + normalization_type + str(sigma_smoothing) + str(sigma_gauss_norm) + str(mean_subtracted_map) + '_' + str(time_window) + '_window_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm' + str(len(elevations)) + '_elevs.npy'

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
        x_mono = x_mono[:, :, elevations, :]
        y_mono = y_mono[:, :, elevations]
        x_mono_mean = x_mono_mean[:, :, elevations, :]
        y_mono_mean = y_mono_mean[:, :, elevations]
        x_bin = x_bin[:, :, elevations, :]
        y_bin = y_bin[:, :, elevations]
        x_bin_mean = x_bin_mean[:, :, elevations, :]
        y_bin_mean = y_bin_mean[:, :, elevations]

        # save regression values for later usage
        coeff_ms = np.zeros((4, participant_numbers.shape[0]))
        coeff_bs = np.zeros((4, participant_numbers.shape[0]))
        scores = np.zeros((4, participant_numbers.shape[0]))

        # plot regression line for each participant
        for i_par, par in enumerate(participant_numbers):
            coeff_ms[0,i_par],coeff_bs[0,i_par],scores[0,i_par] =get_regression_values(x_mono[i_par],y_mono[i_par])
            coeff_ms[1,i_par],coeff_bs[1,i_par],scores[1,i_par] =get_regression_values(x_mono_mean[i_par],y_mono_mean[i_par])
            coeff_ms[2,i_par],coeff_bs[2,i_par],scores[2,i_par] =get_regression_values(x_bin[i_par],y_bin[i_par])
            coeff_ms[3,i_par],coeff_bs[3,i_par],scores[3,i_par] =get_regression_values(x_bin_mean[i_par],y_bin_mean[i_par])

            # sns.set_palette('muted')
            # my_pal = sns.color_palette("hls", 8)
        fig = plt.figure(figsize=(20, 5))
        my_pal = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
                  (0.5058823529411764, 0.4470588235294118, 0.6980392156862745), (0.8, 0.7254901960784313, 0.4549019607843137), (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]

        ax = fig.add_subplot(1, 3, 1)
        ax.set_ylabel('Gain')
        sns.boxplot(data=coeff_ms.T, showfliers=True, palette=my_pal, ax=ax, linewidth=3)
        ax.set_xticklabels(['Mono', 'Mono\n-Mean', 'Bin', 'Bin\n-Mean'])

        ax = fig.add_subplot(1, 3, 2)
        ax.set_ylabel('Bias')
        sns.boxplot(data=coeff_bs.T, showfliers=True, palette=my_pal, ax=ax, linewidth=3)
        ax.set_xticklabels(['Mono', 'Mono\n-Mean', 'Bin', 'Bin\n-Mean'])

        ax = fig.add_subplot(1, 3, 3)
        ax.set_ylabel('Score')
        sns.boxplot(data=scores.T, showfliers=True, palette=my_pal, ax=ax, linewidth=3)
        ax.set_xticklabels(['Mono', 'Mono\n-Mean', 'Bin', 'Bin\n-Mean'])

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig((fig_save_path / (exp_name + '_localization.' + save_type)).as_posix(), dpi=300)

        plt.show()
    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
