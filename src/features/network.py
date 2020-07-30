import numpy as np

from scipy.ndimage import gaussian_filter1d, convolve1d


class Network():
    "This class defines the network, its structure and parameters"

    def __init__(self, dt=0.0001, T=0.3, tau=0.005, learning_rate=0.00005, freq_bands=128, n_elevations=25, gauss_sigma=3):

        ##### Set parameters #####
        self.dt = dt    # time steps
        self.T = T      # run time
        self.ts = np.linspace(0, T, int(T / dt) + 1)    # time steps
        if len(self.ts) == 0:    # that probably means that we want to calculate the steady state response
            self.ts = np.arange(0,2)     # set it to 2 so that the model runs for one step
        self.tau = tau  # tau used for all neuron ode_p_sum
        self.freq_bands = freq_bands    # frequency bands used
        self.n_elevations = n_elevations  # number of elevations used
        self.w = np.random.random_sample((n_elevations, freq_bands)) * 0.1    # initialize the weights
        self.gauss_sigma = gauss_sigma  # Defines the sigma for the input gauss kernel
        self.learning_rate = learning_rate  # learning rate for weights

    def run(self, in_ipsi, in_contra, ele, train=False):
        """ This function runs the model for given time steps time_steps with given ipsi and contralateral inputs.
            ele is the current given elevation. If train is true, a connection matrix w is learned between the last two layers
            If time_steps is not given, the default time steps given at initialization stage are executed"""

        ##### Define variables to store values in #####
        # visual guidance signal
        v_in = np.zeros((self.n_elevations, 1))
        v_in[ele] = 1  # the visual guidance signal is always where the sound comes from

        p_in_c = np.zeros((len(self.ts), self.freq_bands))
        p_in_i = np.zeros((len(self.ts), self.freq_bands))

        r_in_c = np.zeros((len(self.ts), self.freq_bands))
        r_in_i = np.zeros((len(self.ts), self.freq_bands))

        p_sum_i = np.zeros((len(self.ts), self.freq_bands))
        p_sum_c = np.zeros((len(self.ts), self.freq_bands))
        r_ipsi = np.zeros((len(self.ts), self.freq_bands))
        q_ele = np.zeros((len(self.ts), self.n_elevations))

        # define the gauss kernel for convolution
        gauss_kernel = self.gauss(np.arange(-4 * self.gauss_sigma, 4 * self.gauss_sigma), 0, self.gauss_sigma)

        # since the input does not change over time. We can do this calculation ouside the loop
        excitatory_in_p_i = convolve1d(self.out_thres(in_ipsi), weights=gauss_kernel, axis=0, mode='reflect')
        excitatory_in_p_c = convolve1d(self.out_thres(in_contra), weights=gauss_kernel, axis=0, mode='reflect')

        # run network for given time steps
        for t in range(0, len(self.ts) - 1):
            # p_In_ipsi neuron
            # feed inputs ipsi inhibition
            p_in_i[t + 1, :] = p_in_i[t, :] + self.dt * self.ode_p_in(p_in_i[t, :], excitatory_in_p_i)

            # r_In_ipsi neuron
            excitatory_in = self.out_thres(in_ipsi)
            inhibitory_in = self.out_thres(p_in_i[t, :])
            r_in_i[t + 1, :] = r_in_i[t, :] + self.dt * self.ode_r_in(r_in_i[t, :], excitatory_in, inhibitory_in)

            # p_In_contra neuron
            # feed inputs ipsi inhibition
            p_in_c[t + 1, :] = p_in_c[t, :] + self.dt * self.ode_p_in(p_in_c[t, :], excitatory_in_p_c)

            # r_In_contra neuron
            excitatory_in = self.out_thres(in_contra)
            inhibitory_in = self.out_thres(p_in_c[t, :])
            r_in_c[t + 1, :] = r_in_c[t, :] + self.dt * self.ode_r_in(r_in_c[t, :], excitatory_in, inhibitory_in)

            # p_sum neurons
            excitatory_in = self.out_thres(r_in_i[t, :])
            p_sum_i[t + 1, :] = p_sum_i[t, :] + self.dt * self.ode_p_sum(p_sum_i[t, :], excitatory_in)

            excitatory_in = self.out_thres(r_in_c[t, :])
            p_sum_c[t + 1, :] = p_sum_c[t, :] + self.dt * self.ode_p_sum(p_sum_c[t, :], excitatory_in)

            # r_ipsi neuron
            excitatory_in = self.out_thres(r_in_i[t, :])
            inhibitory_in = self.out_thres(p_sum_c[t, :]) + self.out_thres(p_sum_i[t, :])
            r_ipsi[t + 1, :] = r_ipsi[t, :] + self.dt * self.ode_r(r_ipsi[t, :], excitatory_in, inhibitory_in)

            # q readout neurons
            excitatory_in = np.dot(self.out_thres(r_ipsi[t + 1, :]), self.w.T)
            q_ele[t + 1, :] = q_ele[t, :] + self.dt * self.ode_q_sum(q_ele[t, :], excitatory_in)

            if train:
                # Learning
                q_ = q_ele[t+1, :, np.newaxis]
                r_ = r_ipsi[t+1, :, np.newaxis]
                v_ = v_in
                # Works, but is supervised ...
                self.w[:, :] = self.w[:, :] + self.learning_rate * (r_.T - self.w[:, :]) * v_
        return q_ele, r_ipsi, self.w

    ##### Define all the neuron ODE functions #####
    # Defines the output transfer

    def out_thres(self, q, threshold=0.0, slope=1):
        return np.minimum(np.maximum((q - threshold) * slope, 0), 1)

    # define a gauss function

    def gauss(self, x, mean, sigma):
        if sigma == 0.0:
            return np.zeros(x.shape)
        else:
            tmp = np.exp(-(x - mean)**2 / (2 * sigma**2))
            return tmp / np.max(tmp)

    # define the ODE for inhibitory input neurons
    def ode_p_in(self, p, excitatory_in):
        # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
        alpha = 1
        # beta defines the upper limit of the membrane potential
        beta = 1

        # calculate the change of r_Alearn
        d_p = -alpha * p + (beta - p) * excitatory_in

        return d_p / self.tau

    # define the ODE for gaussian filter neurons

    def ode_r_in(self, r, excitatory_in, inhibitory_in):
        # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
        alpha = 1
        # beta defines the upper limit of the membrane potential
        beta = 200
        # gamma defines the subtractive influence of the inhibitory input
        gamma = 0.0
        # kappa defines the divisive influence of the inhibitory input
        kappa = 200

        # calculate the change of r_Alearn
        d_r = -alpha * r * excitatory_in + (beta - r) * excitatory_in - (gamma + kappa * r) * inhibitory_in

        return d_r / self.tau

    # define the ODE for neuron p_sum

    def ode_p_sum(self, p, excitatory_in):
        # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
        alpha = 1
        # beta defines the upper limit of the membrane potential
        beta = 1

        # calculate the change of r_Alearn
        d_p = -alpha * p + (beta - p) * excitatory_in

        return d_p / self.tau

    # define the ODE for integration neurons

    def ode_r(self, r, excitatory_in, inhibitory_in=0):
        # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
        alpha = 1
        # beta defines the upper limit of the membrane potential
        beta = 2
        # gamma defines the subtractive influence of the inhibitory input
        gamma = 0
        # kappa defines the divisive influence of the inhibitory input
        kappa = 1

        # calculate the change of r_Alearn
        d_r = -alpha * r * excitatory_in + (beta - r) * excitatory_in - (gamma + kappa * r) * inhibitory_in

        return d_r / self.tau

    # define the ODE for read out neurons

    def ode_q_sum(self, q, excitatory_in):
        # alpha defines the decay rate of the membrane potential but also the value to which it saturates (implicitly)
        alpha = 1
        # beta defines the upper limit of the membrane potential
        beta = 1

        # calculate the change of r_Alearn
        d_q = -alpha * q + (beta - q) * excitatory_in

        return d_q / self.tau
