import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch
from itertools import islice
import matplotlib as mpl
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


class LinearReg():

    def __init__(self, x, y):
        from sklearn.linear_model import LinearRegression

        self.lr_model = LinearRegression()

        self.x = np.squeeze(x[:, 1]).reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        self.lr_model.fit(self.x, self.y)

        self.rr = self.lr_model.score(self.x, self.y)

    def get_fitted_line(self):
        return [self.x, self.lr_model.predict(self.x)]

    def get_coefficients(self):
        return self.lr_model.coef_[0, 0], self.lr_model.intercept_[0]

    def get_score(self, x=0, y=0):
        if x == 0 or y == 0:
            return self.rr
        else:
            return self.lr_model.score(x, y)

    def print_coefficients(self):
        print('Gain: {0:1.2f}, Bias: {1:1.2f}, , r^2: {2:1.2f}'.format(self.lr_model.coef_[0, 0], self.lr_model.intercept_[0], self.rr))
        return ('Gain: {0:1.2f},\nBias: {1:1.2f},\n'+r'$r^2$: {2:1.2f}').format(self.lr_model.coef_[0, 0], self.lr_model.intercept_[0], self.rr)


def localize_sound(psd_all, data_to_compare, metric='correlation'):

    x_test = np.zeros((psd_all.shape[0], psd_all.shape[1], 2))
    y_test = np.zeros((psd_all.shape[0], psd_all.shape[1]))

    # walk over all samples
    for i in range(psd_all.shape[0]):
        for ii in range(psd_all.shape[1]):
            data_point = np.squeeze(psd_all[i, ii, :])
            data_point = data_point[np.newaxis]

            # search for smallest distance between white noise data points and the randomly choosen sound sample
            dists = distance.cdist(data_to_compare, data_point, metric=metric)
            minimal_dist_ind = np.argmin(dists)
            y_test[i, ii] = minimal_dist_ind

            # save the data points
            x_test[i, ii, 0] = i
            x_test[i, ii, 1] = ii

    return x_test, y_test


def get_localization_coefficients_score(x_test, y_test):
    x_test = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], 2))
    y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))

    lr_model = LinearReg(x_test, y_test)
    gain, bias = lr_model.get_coefficients()
    return gain, bias, lr_model.get_score()


def removeOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return np.array(resultList)


def scale_v(x_test, y_test):
    a = x_test[:, :, 1]/np.max(x_test[:, :, 1])
    a = a*135 - 45
    x_test[:, :, 1] = a

    a = y_test[:, :]/np.max(y_test[:, :])
    a = a*135 - 45
    y_test[:, :] = a

    return x_test, y_test


def plot_localization_result(x_test, y_test, ax, sound_files, scale_values=False, linear_reg=True, scatter_data=True):
    n_sound_types = len(sound_files)

    if scale_values:
        x_test, y_test = scale_v(x_test, y_test)

    x_test = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], 2))
    y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))

    ax.plot(np.arange(np.ceil(np.min(x_test)), np.ceil(np.max(x_test))), np.arange(np.ceil(np.min(x_test)), np.ceil(np.max(x_test))), color='grey', linestyle='--', alpha=0.3)

    error_mse = 0
    for i in range(0, n_sound_types):
        # get the data points, NOT the sound file type
        x = x_test[x_test[:, 0] == i, 1]
        y = y_test[x_test[:, 0] == i]

        if scatter_data:
            ax.scatter(x, y, s=(n_sound_types / (i + 1)) * 20, alpha=0.85, label=sound_files[i].name.split('.')[0])
        error_mse += ((y - x) ** 2).mean(axis=0)

    error_mse /= n_sound_types

    if linear_reg:
        lr_model = LinearReg(x_test, y_test)
        [x, y] = lr_model.get_fitted_line()
        ax.plot(x, y, color='black')
        text_str = lr_model.print_coefficients()
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, verticalalignment='top', bbox=props)

    # ax.set_xlabel('True Elevation')
    # ax.set_ylabel('Elevation based on WN')

    # print("MSE : {0:1.2f}".format(error_mse))
    return ax


def set_layout(drawing_size=25, regular_seaborn=False, box_frame=True):
    if regular_seaborn:
        import seaborn as sns
        sns.set(color_codes=True)
        plt.style.use('seaborn-whitegrid')
        # sns.set_context("poster")
        # sns.set(font="Arial")

        # plt.style.use('seaborn')
        # sns.set_style("ticks")

    mpl.rcParams['grid.linestyle'] = ':'

    mpl.rcParams['font.size'] = drawing_size
    mpl.rcParams['font.style'] = 'normal'
    mpl.rcParams['font.weight'] = 'heavy'
    # mpl.rcParams['font.family'] = ['Symbol']

    mpl.rcParams['figure.titlesize'] = int(drawing_size * 1.3)
    mpl.rcParams['figure.titleweight'] = 'heavy'

    mpl.rcParams['lines.linewidth'] = int(drawing_size / 5)

    mpl.rcParams['axes.labelsize'] = drawing_size
    mpl.rcParams['axes.labelweight'] = 'heavy'
    mpl.rcParams['axes.titlesize'] = int(drawing_size * 1.3)
    mpl.rcParams['axes.titleweight'] = 'heavy'
    mpl.rcParams['xtick.labelsize'] = int(drawing_size * 1)
    mpl.rcParams['ytick.labelsize'] = int(drawing_size * 1)

    if box_frame:
        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['legend.fontsize'] = int(drawing_size * 0.9)
        mpl.rcParams['legend.frameon'] = True
        mpl.rcParams['legend.framealpha'] = 0.5
    else:
        mpl.rcParams['legend.fancybox'] = False
        mpl.rcParams['legend.fontsize'] = int(drawing_size * 0.9)
        mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.facecolor'] = 'inherit'
    mpl.rcParams['legend.edgecolor'] = '0.8'

    mpl.rcParams['figure.figsize'] = [20.0, 10.0]
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 400
    mpl.rcParams['image.cmap'] = 'viridis'


def mesh_plot(data):
    # fig is the figure handler
    # data is a 2d array
    # returns the axis of the figure

    # define x and y axis
    x = np.linspace(np.min(data[0, :]), np.max(data[0, :]), data.shape[0])
    y = np.linspace(np.min(data[1, :]), np.max(data[1, :]), data.shape[1])
    # x = np.linspace(0, 1, data.shape[0])
    # y = np.linspace(0, 1, data.shape[1])
    x, y = np.meshgrid(x, y)
    # create figure with 3d subplot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # plot data with nice colors
    ax.plot_surface(
        x,
        y,
        data.T,
        rstride=1,
        cstride=1,
        cmap=cm.jet,
        shade=True,
        linewidth=20)

    # c = ax.contour(x , y , tmp.T, colors='black', zorder=0)
    # # Thicken the zero contour.
    # zc = c.collections[:]
    # plt.setp(zc, linewidth=4)
    return ax


def set_correct_axis_3D(ax, cutoff_freq_low, cutoff_freq_high):
    ax.set_ylabel('Frequencies (kHz)')
    ticks, _ = plt.yticks()
    ax.set_yticklabels(
        [str(int(k / 1000)) for k in np.linspace(cutoff_freq_low, cutoff_freq_high, len(ticks) + 1)])

    ax.set_zlabel('SPL (au)')
    ax.set_xlabel('Elevation')
    ticks, _ = plt.xticks()
    ax.set_xticklabels([str(int(k)) for k in np.linspace(-45, 90, len(ticks))])


def set_correct_axis_2D(ax, cutoff_freq_low, cutoff_freq_high):
    ax.set_xlabel('Frequencies (kHz)')
    ticks, _ = plt.xticks()
    ax.set_xticklabels([str(int(k / 1000)) for k in np.linspace(cutoff_freq_low, cutoff_freq_high, len(ticks) + 1)])

    ax.set_ylabel('Elevation')
    ticks, _ = plt.yticks()
    ax.set_yticklabels([str(int(k)) for k in np.linspace(-45, 90, len(ticks))])


def plot_elevation_map(data, sounds, figsize=(25, 5)):
    fig = plt.figure(figsize=figsize)

    for i in range(0, data.shape[0], 1):
        ax = fig.add_subplot(1, data.shape[0], 1 + i)
        ax.set_title(sounds[i].split('/')[1].split('_')[0].split('.')[0])
        data_revised_i = np.squeeze(data[i, :, :])

        c = ax.pcolormesh(data_revised_i[:, :])

        plt.colorbar(c)


def filter_dataset(dataset, normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=0):
    # applies filter and normalization on dataset. normalization_type can be one of the following: 'sum_1','l2','l1'
    ds = np.copy(dataset)
    # first do a smoothing
    if sigma_smoothing > 0:
        ds = gaussian_filter1d(ds, sigma=sigma_smoothing, mode='nearest', axis=2)

    # now normalize the data
    if normalization_type.find('sum_1') >= 0:
        # print('Sum 1 Normalization')
        ds = (ds / np.transpose(np.tile(np.sum(ds, axis=2), [ds.shape[2], 1, 1]), [1, 2, 0]))
    elif normalization_type.find('l1') >= 0:
        # print('L1 Normalization')
        ds = (ds - np.transpose(np.tile(np.min(ds, axis=2), [ds.shape[2], 1, 1]), [1, 2, 0]))
        ds = (ds / np.transpose(np.tile(np.max(ds, axis=2), [ds.shape[2], 1, 1]), [1, 2, 0]))
    elif normalization_type.find('l2') >= 0:
        # print('L2 Normalization')
        ds = (ds / np.transpose(np.tile(np.sqrt(np.sum(ds ** 2, axis=2)), [ds.shape[2], 1, 1]), [1, 2, 0]))
    else:
        # print('No Normalization')
        ds = ds

        # now divide the data by a gaussian convolved version of it (filter it)
    if sigma_gauss_norm > 0:
        ds = ds / (gaussian_filter1d(ds, sigma=sigma_gauss_norm, mode='nearest', axis=2))

    return ds


def set_axis(ax, label=False):
    if label:
        ax.set_xlabel('True Elevation [deg]')
    t = np.zeros(6)

    t[0] = -55
    t[1] = -45
    t[2] = 0
    t[3] = 45
    t[4] = 90
    t[5] = 100
    ax.set_xticks(t[1:-1])

    t = np.zeros(6)
    t[0] = -55
    t[1] = -45
    t[2] = 0
    t[3] = 45
    t[4] = 90
    t[5] = 100
    ax.set_yticks(t[1:-1])

    return ax


def set_axis_all_elevations(ax, label=False):
    if label:
        ax.set_xlabel('True Elevation [deg]')

    t = np.zeros(6)
    t[0] = -55
    t[1] = -45
    t[2] = -45+91*0
    t[3] = -45+91*1
    t[4] = -45+91*2
    t[5] = -45+91*3
    ax.set_xticklabels(t[1:])

    # t = np.zeros(6)
    # t[0] = -55
    # t[1] = -45
    # t[2] = 0
    # t[3] = 45
    # t[4] = 90
    # t[5] = 100

    return ax
