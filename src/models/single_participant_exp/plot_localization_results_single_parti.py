import numpy as np
import matplotlib.pyplot as plt
import gammatone_read_cipic as file_reader
import helpers as hp

hp.set_layout(drawing_size=20, regular_seaborn=True)


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


# read the data
file_reader = file_reader.FileReader(44100)
sound_files = file_reader.files
psd_all_c, psd_all_i = file_reader.read_files(freq_bands, participant_number, snr, normalize, azimuth, time_window)


# filter the data
psd_all_c = hp.filter_dataset(psd_all_c, normalization_type=normalization_type, sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)
psd_all_i = hp.filter_dataset(psd_all_i, normalization_type=normalization_type, sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)

psd_binaural = hp.filter_dataset(psd_all_i / psd_all_c, normalization_type=normalization_type, sigma_smoothing=0, sigma_gauss_norm=0)

map = psd_binaural
if mean_subtracted_map:
    learned_map = np.mean(map, axis=0)
    mean_learned_map = np.mean(learned_map, axis=0)
    learned_map = learned_map - mean_learned_map
else:
    learned_map = np.mean(map, axis=0)
    learned_map = learned_map


# psd_all__tmp, psd_all_i = file_reader.read_files(freq_bands, 3, snr, normalize, azimuth, time_window, False)
# psd_all_i = hp.filter_dataset(psd_all_i, normalization_type=normalization_type, sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)
# psd_binaural = hp.filter_dataset(psd_all_i / psd_all_c, normalization_type=normalization_type, sigma_smoothing=0, sigma_gauss_norm=0)


# fig = plt.figure(figsize=(5, 5))
# plt.pcolormesh(learned_map)


# hp.plot_elevation_map(psd_all_i,sound_files)

# %%

fig = plt.figure(figsize=(20, 5))
# plt.suptitle('Single Participant')
# Monoaural Data (Ipsilateral), No Mean Subtracted
ax = fig.add_subplot(1, 4, 1)
x_test, y_test = hp.localize_sound(psd_all_i, learned_map)
hp.plot_localization_result(x_test, y_test, ax, sound_files, scale_values=True, linear_reg=True)
ax.set_title('Monoaural')
hp.set_axis(ax)
ax.set_ylabel('Estimated Elevation [deg]')


# Monoaural Data (Ipsilateral), Mean Subtracted
ax = fig.add_subplot(1, 4, 2)
# create mean subtracted data
psd_all_i_mean = psd_all_i - np.transpose(np.tile(np.mean(psd_all_i, axis=1), [psd_all_i.shape[1], 1, 1]), [1, 0, 2])

x_test, y_test = hp.localize_sound(psd_all_i_mean, learned_map)
hp.plot_localization_result(x_test, y_test, ax, sound_files, scale_values=True, linear_reg=True)
ax.set_title('Mono - Mean')
hp.set_axis(ax)


# Binaural Data (Ipsilateral), No Mean Subtracted
ax = fig.add_subplot(1, 4, 3)
x_test, y_test = hp.localize_sound(psd_binaural, learned_map)
hp.plot_localization_result(x_test, y_test, ax, sound_files, scale_values=True, linear_reg=True)
ax.set_title('Binaural')
hp.set_axis(ax)


# Binaural Data (Ipsilateral), Mean Subtracted
ax = fig.add_subplot(1, 4, 4)
# ax.axis(xmin=-45, xmax=90, ymin=-45 ,ymax=90,option='equal')

psd_binaural_mean = psd_binaural - np.transpose(np.tile(np.mean(psd_binaural, axis=1), [psd_binaural.shape[1], 1, 1]), [1, 0, 2])
x_test, y_test = hp.localize_sound(psd_binaural_mean, learned_map)

hp.plot_localization_result(x_test, y_test, ax, sound_files, scale_values=True, linear_reg=True)
ax.set_title('Bin - Mean')
hp.set_axis(ax)

# plt.savefig('./figures/localization_single_participant.pdf', dpi=300)

plt.show()
