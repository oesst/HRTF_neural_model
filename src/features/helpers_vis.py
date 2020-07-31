import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.features.filters as filters


# Define some colors
C0 = [31 / 255, 119 / 255, 180 / 255]
C1 = [255 / 255, 127 / 255, 14 / 255]
C2 = [44 / 255, 160 / 255, 44 / 255]
C3 = [214 / 255, 39 / 255, 40 / 255]


MY_COLORS = [C0, C1, C2, C3]


class LinearReg():

    def __init__(self, x, y):
        from sklearn.linear_model import LinearRegression

        self.lr_model = LinearRegression()

        self.x = x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        self.lr_model.fit(self.x, self.y)

        self.rr = self.lr_model.score(self.x, self.y)

    def get_fitted_line(self):
        return [self.x, self.lr_model.predict(self.x)]

    def get_coefficients(self):
        # gain, bias
        return self.lr_model.coef_[0, 0], self.lr_model.intercept_[0]

    def get_score(self, x=0, y=0):
        if x == 0 or y == 0:
            return self.rr
        else:
            return self.lr_model.score(x, y)

    def print_coefficients(self):
        print('Gain: {0:1.2f}, Bias: {1:1.2f}, , r^2: {2:1.2f}'.format(self.lr_model.coef_[0, 0], self.lr_model.intercept_[0], self.rr))
        return ('Gain: {0:1.2f},\nBias: {1:1.2f},\n' + r'$r^2$: {2:1.2f}').format(self.lr_model.coef_[0, 0], self.lr_model.intercept_[0], self.rr)


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


def scale_v(x_test, y_test, n_elevations):
    a = x_test[:, :, 1] / n_elevations
    a = a * (n_elevations - 1) * 5.625 - 45
    x_test[:, :, 1] = a

    a = y_test[:, :] / n_elevations
    a = a * (n_elevations - 1) * 5.625 - 45
    y_test[:, :] = a

    return x_test, y_test


def plot_localization_result(x_test, y_test, ax, sound_files, scale_values=False, linear_reg=True, disp_values=False, scatter_data=True, reg_color=""):
    n_sound_types = len(sound_files)
    if scale_values:
        x_test, y_test = scale_v(x_test, y_test, x_test.shape[0])

    x_test = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], 2))
    y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))

    ax.plot(np.arange(np.ceil(np.min(x_test)), np.ceil(np.max(x_test))), np.arange(
        np.ceil(np.min(x_test)), np.ceil(np.max(x_test))), color='grey', linestyle='--', alpha=0.3, label='_nolegend_')

    # error_mse = 0
    for i in range(0, n_sound_types):
        # get the data points, NOT the sound file type
        x = x_test[x_test[:, 0] == i, 1]
        y = y_test[x_test[:, 0] == i]

        if scatter_data:
            ax.scatter(x, y, s=(n_sound_types / (i + 1)) * len(sound_files), alpha=0.85, label=sound_files[i].name.split('.')[0])
        # error_mse += ((y - x) ** 2).mean(axis=0)

    # error_mse /= n_sound_types

    if linear_reg:
        lr_model = LinearReg(x_test, y_test)
        [x, y] = lr_model.get_fitted_line()

        if scatter_data:
            ax.plot(x, y, color='black')
        else:
            if len(reg_color) > 0:
                ax.plot(x, y, alpha=0.6, color=reg_color)
            else:
                ax.plot(x, y, alpha=0.6)

        if disp_values:
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
    import seaborn as sns
    # 21 defines the number of sound types
    # sns.set_palette(sns.color_palette("husl", 21))
    mpl.rcParams['grid.linestyle'] = ':'

    mpl.rcParams['font.size'] = drawing_size
    mpl.rcParams['font.style'] = 'normal'
    # mpl.rcParams['font.family'] = ['Symbol']

    mpl.rcParams['figure.titlesize'] = int(drawing_size * 1.3)

    mpl.rcParams['lines.linewidth'] = int(drawing_size / 5)

    mpl.rcParams['axes.labelsize'] = drawing_size
    mpl.rcParams['axes.titlesize'] = int(drawing_size * 1.3)
    mpl.rcParams['xtick.labelsize'] = int(drawing_size * 1)
    mpl.rcParams['ytick.labelsize'] = int(drawing_size * 1)

    if box_frame:
        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['legend.fontsize'] = int(drawing_size * 1)
        mpl.rcParams['legend.frameon'] = True
        mpl.rcParams['legend.framealpha'] = 0.5
    else:
        mpl.rcParams['legend.fancybox'] = False
        mpl.rcParams['legend.fontsize'] = int(drawing_size * 1)
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


def set_axis(ax, n_elevations=25, label=False):
    if label:
        ax.set_xlabel('True Elevation [deg]')
    t = np.zeros(6)

    if n_elevations == 25:
        t[0] = -55
        t[1] = -45
        t[2] = 0
        t[3] = 45
        t[4] = 90
        t[5] = 100
        ax.set_xticks(t[1:-1])
        ax.set_yticks(t[1:-1])
    elif n_elevations == 50:
        t = np.zeros(9)

        t[0] = -55
        t[1] = -45 + 45 * 0
        t[2] = -45 + 45 * 1
        t[3] = -45 + 45 * 2
        t[4] = -45 + 45 * 3
        t[5] = -45 + 45 * 4
        t[6] = -45 + 45 * 5
        t[7] = -45 + 45 * 6
        t[8] = -45 + 45 * 7
        ax.set_xticks(t[1:-1])
        ax.set_yticks(t[1:-1])
    else:
        t[0] = -55
        t[1] = -45
        t[2] = -45 + ((n_elevations * 5.625) / 2.5) * 1
        t[3] = -45 + ((n_elevations * 5.625) / 2.5) * 2
        t[4] = -45 + ((n_elevations * 5.625) / 2.5) * 3
        t[5] = -45 + ((n_elevations * 5.625) / 2.5) * 4
        ax.set_xticks(t[1:-1])
        ax.set_yticks(t[1:-1])
    return ax


# def set_axis_all_elevations(ax, label=False):
#     if label:
#         ax.set_xlabel('True Elevation [deg]')
#
#
#
#     return ax


class ERBFormatter(mpl.ticker.EngFormatter):
    """
    Axis formatter for gammatone filterbank analysis. This formatter calculates
    the ERB spaced frequencies used for analysis, and renders them similarly to
    the engineering axis formatter.
    The scale is changed so that `[0, 1]` corresponds to ERB spaced frequencies
    from ``high_freq`` to ``low_freq`` (note the reversal). It should be used
    with ``imshow`` where the ``extent`` argument is ``[a, b, 1, 0]`` (again,
    note the inversion).
    """

    def __init__(self, low_freq, high_freq, *args, **kwargs):
        """
        Creates a new :class ERBFormatter: for use with ``matplotlib`` plots.
        Note that this class does not supply the ``units`` or ``places``
        arguments; typically these would be ``'Hz'`` and ``0``.
        :param low_freq: the low end of the gammatone filterbank frequency range
        :param high_freq: the high end of the gammatone filterbank frequency
          range
        """
        self.low_freq = high_freq
        self.high_freq = low_freq
        super().__init__(*args, **kwargs)

    def _erb_axis_scale(self, fraction):
        return filters.erb_point(self.low_freq, self.high_freq, fraction)

    def __call__(self, val, pos=None):
        newval = self._erb_axis_scale(val)
        return super().__call__(newval, pos)
