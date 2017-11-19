import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as st

from telepyth import TelepythClient

MARKERS = ['o', 'v','s', '*', 'D', 'P', '^']


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def set_color(method):
    cmap = matplotlib.cm.get_cmap('tab10')
    for i, t in enumerate(APPROX_TYPES):
        if method == t:
            break
    return cmap(i)


def set_marker(method):
    for i, t in enumerate(APPROX_TYPES):
        if method == t:
            break
    return MARKERS[i]


def save_figure_data(dataset_name, kernel_name, params, data, errs=True):
    if errs:
        t = 'errs'
    else:
        t = 'mses'
    filename = 'figure_data/ka/Halton' + t + '_'.join([dataset_name, kernel_name] + \
               ['%r' % p for p in params])
    np.savez(filename, data)


def plot_approx_errors(errs, d, dataset_name, kernel_type, params,
                       semilogy=False, mse=False):
    start_deg, max_deg, _, shift, step, _ = params
    tp = TelepythClient()
    fig = plt.figure()
    ax = fig.add_subplot('111')
    x = 2 * (np.arange(start_deg, max_deg+step, step) + shift) * (d + 1)
    ci = np.empty((errs.shape[0], errs.shape[1], 2))
    mean = np.zeros((errs.shape[0], errs.shape[1]))

    for i in range(errs.shape[0]-1, -1, -1):
        color = set_color(APPROX_TYPES[i])
        m = set_marker(APPROX_TYPES[i])
        if errs[i, 0, 0] == 0.:
            continue
        for j in range(errs[i].shape[0]):
            mean[i, j], ci[i, j, 0], ci[i, j, 1] = \
                mean_confidence_interval(errs[i,j,:])
        if semilogy:
            ax.semilogy(x, mean[i], '.-', label=APPROX_TYPES[i], color=color,
                        marker=m, markersize=5)
            ax.fill_between(x, np.maximum(0, ci[i, :, 0]), ci[i, :, 1],
                            alpha=0.3, color=color)
        else:
            ax.plot(x, mean[i], '.-', label=APPROX_TYPES[i], color=color,
                    marker=m, markersize=5)
            ax.fill_between(x, np.maximum(0, ci[i, :, 0]), ci[i, :, 1],
                            alpha=0.3, color=color)

    ylabel = r'$\frac{\|K - \hat{K}\|}{\|K\|}$'
    if mse == True:
        ylabel = 'MSE'
    ax.set_xlabel('$n$', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    title = '%s (%s)' % (kernel_type, dataset_name)
    fig.suptitle(title, fontsize=16)
    ax.legend(loc='best', framealpha=0.1)
    plt.tight_layout(pad=3)
    filename = '_'.join([dataset_name, kernel_type] + \
               ['%r' % p for p in params])
    if semilogy:
        fig_type = 'semilogy'
    else:
        fig_type = 'plain'
    format = 'pdf'
    plt.savefig('figure_data/errs%s_%s.%s' % (filename, fig_type, format),
                dpi=1200, format=format)
    plt.show()
    tp.send_figure(fig, title)
