import numpy as np
import scipy.stats as sp
from matplotlib import pyplot as plt
from numpy.random import normal


class Simulations:
    pass


def multivariate_brownian(time_increment, drift, cov, sigma, periods_ahead, num_simulations, s0):
    # periods_ahead = how many projections (simulations) in time ahead you want
    # num_simulations = the number of times you want to project a random path starting_values = in the formula for "dx",
    # that is the change in price going forward, the latest available observation

    zeroMu = np.zeros(len(cov))
    dt = time_increment
    dx = np.zeros((num_simulations, 1 + periods_ahead, len(cov)))
    dx[:, 0, :] = tuple([1] * len(cov))

    for i in range(num_simulations):
        for j in range(1, 1 + periods_ahead):
            dx[i, j, :] = np.exp(
                (drift - sigma / 2) * dt + np.random.multivariate_normal(zeroMu, cov) * np.sqrt(sigma) * np.sqrt(dt))

        dx[i] = s0 * dx[i].cumprod(axis=0)

    sims_per_asset = np.zeros((len(cov), 1 + periods_ahead, num_simulations))

    for j in range(0, len(cov)):
        sims_per_asset[j, :, :] = dx[:, :, j].T

    dxOut = np.zeros(((1 + periods_ahead) * num_simulations, dx.shape[2], 1))

    # Reshape the 3D-matrix into 2D
    for i in range(dx.shape[2]):
        dxOut[:, i] = dx[:, :, i].reshape((1 + periods_ahead) * num_simulations, 1)
    dxOut = np.squeeze(dxOut)

    return sims_per_asset, dxOut


def vasicek(starting_value, mean_reversion, mu, sigma, time_increment, num_simulations, periods_ahead):
    a = mean_reversion
    b = mu
    rt = starting_value
    dt = time_increment
    dx = np.zeros(num_simulations, 1 + periods_ahead)

    for i in range(num_simulations):
        for j in range(1, 1 + periods_ahead):
            dx[i, j] = a(b - rt) * dt + np.sqrt(sigma) * np.random.normal(0, 1) * np.sqrt(dt)

    return dx


def kernel_density_estimation(data):
    # bw_methods = 'scott', 'silverman' or scalar

    bins = np.int(np.round(len(data) * 0.02))
    pdf = sp.gaussian_kde(data, bw_method='scott')
    x = np.linspace(np.min(data), np.max(data), num=50)
    z = pdf.evaluate(x)

    plt.figure()
    plt.hist(data, bins=bins, density=True)
    plt.plot(x, z, 'r')

    return data, pdf, x, z


def optimal_bandwidth(data):
    n = len(data)
    std = np.std(data)
    iqr = sp.iqr(data)

    return 0.9 * np.min([std, iqr / 1.34]) * n ** (-1 / 5)


def monte_carlo_from_pdf(pdf, num_simulations):
    # pdf = probability density function of a data set
    # num_simulations = the number of times we extract values (re-sampling) and form a new pdf, this is the Monte Carlo part

    simulated_data = np.squeeze(pdf.resample(num_simulations, seed=None))
    bins = np.int(np.round(len(simulated_data) * 0.02))

    return simulated_data, bins