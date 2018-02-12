import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import scipy.stats as st

from matplotlib.ticker import LogFormatterMathtext, MaxNLocator, \
                              ScalarFormatter
from os.path import join

from dataset import PARAMS, DIMS


CMAP_I = {'G': 2, 'Gort':4, 'ROM':3, 'QMC':1, 'GQ':9, 'B':0, 'H':8}
MARKERS = {'G':'s', 'Gort':'D','ROM':'*', 'QMC':'v', 'GQ':'^', 'B':'o', 'H':'P'}


#TODO setup layout params to work properly with different number of
#     datasets/kernels.
#TODO setup subplots to work with 1 dataset/kernel.
#layout parameters
top = {True: 0.92 , False: 0.92}
right = {True: 0.92 , False: 0.92}
left = {True: 0.07 , False: 0.07}
title_h = {True: 1.03 , False: 1.15}
legend_h = {True: 2.32 , False: 2.45}
#fontsizes
basefontsize = 12
bigfontsize = 14
#figure sizes
w = 2.2
h = 6


def plot_errors(errs_dic, datasets, kernels, approx_types, semilogy=False,
                acc=False, exact=None, params=None):
    m = len(datasets)
    n = len(kernels)
    fig, axes = plt.subplots(ncols=m, nrows=n, figsize=(w*m, h))

    if acc:
        ylabel = 'accuracy/R^2'
    else:
        ylabel = r'$\frac{\|K - \hat{K}\|}{\|K\|}$'

    for l, dataset_name in enumerate(datasets):
        errs_d = errs_dic[dataset_name]
        if params is None:
                params = PARAMS[dataset_name]
        start_deg, max_deg, _, shift, step, _, _ = params

        x = np.arange(start_deg, max_deg + step, step) + shift
        for k, kernel in enumerate(kernels):
            errs = errs_d[kernel]
            formatter = LogFormatterMathtext()
            formatterx = ScalarFormatter(useMathText=True, useOffset=False)
            formatterx.set_powerlimits((-1,1))
            axes[k, l].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[k, l].yaxis.set_major_locator(MaxNLocator(6))

            ci = np.empty((max_deg, 2))
            m = np.zeros(max_deg)
            if acc:
                axes[k, l].axhline(exact[dataset_name][kernel], linestyle='--',
                                   label=r'\textbf{exact}', color='black')
            for a in approx_types:
                if a in errs:
                    er = errs[a]
                else:
                    continue
                ma = MARKERS[a]
                color = set_color(a)
                for j in range(er.shape[0]):
                    m[j], ci[j, 0], ci[j, 1] = \
                        mean_confidence_interval(er[j,:])
                if semilogy:
                    axes[k, l].yaxis.set_major_formatter(formatter)
                    axes[k, l].semilogy(x, m, '.-', label=r'\textbf{%s}' % a,
                                        color=color, marker=ma, markersize=5)
                    axes[k, l].fill_between(x, np.maximum(0, ci[:, 0]),
                                            ci[:, 1], alpha=0.3, color=color)
                else:
                    axes[k, l].yaxis.set_major_formatter(formatterx)
                    axes[k, l].plot(x, m, '.-', label=r'\textbf{%s}' % a,
                                    color=color, marker=ma, markersize=5)
                    axes[k, l].fill_between(x, np.maximum(0, ci[:, 0]),
                                            ci[:, 1], alpha=0.3, color=color)
            # last row gets x labels
            if k == n - 1:
                axes[k, l].set_xlabel(r'$n$', fontsize=basefontsize)
            # first column gets y labels
            if l == 0:
                axes[k, l].set_ylabel(ylabel, fontsize=bigfontsize)
                if kernel == 'RBF':
                    kernel_ = 'Gaussian'
                else:
                    kernel_ = kernel
                axes[k, l].annotate(r'\textbf{%s}' % kernel_, [0.,0.5],
                                    xytext=(-axes[k, l].yaxis.labelpad, 0),
                                    xycoords=axes[k, l].yaxis.label,
                                    textcoords='offset points',
                                    fontsize=bigfontsize, ha='right',
                                    va='center', rotation=90)
            # first row gets title
            if k == 0:
                title = r'\textbf{%s}' % dataset_name
                axes[k, l].set_title(title, fontsize=basefontsize,
                                     y=title_h[semilogy])
    # authentic organic hand-made legend!
    patches = []
    for a in approx_types:
        patches.append(mlines.Line2D([], [], color=set_color(a),
                                     marker=MARKERS[a], markersize=5,
                                     label=a))
    plt.legend(bbox_to_anchor=(1.05, legend_h[semilogy]),
               loc=2, borderaxespad=0., framealpha=0.0,
               fontsize=basefontsize, handles=patches)
    # makes plot look nice
    fig.tight_layout()
    fig.subplots_adjust(left=left[semilogy], top=top[semilogy],
                        right=right[semilogy])
    plt.show()
    return fig


def plot_time(times, save_to='results'):
    fig = plt.figure()
    for a, t in times.items():
        plt.semilogy(DIMS[1:], t[1:,:].mean(1), label=a,
                     color=set_color(a), marker=MARKERS[a])
        plt.legend(loc='best')
        plt.ylabel('Time, s', fontsize=basefontsize)
        plt.title('Explicit mapping time', fontsize=basefontsize)
        plt.xlabel(r'$d$, dataset input dimension', fontsize=basefontsize)
    fig.tight_layout()
    top = 0.92
    right = 0.99
    left = 0.11
    bottom = 0.11
    fig.subplots_adjust(left=left, top=top, right=right,bottom=bottom)
    plt.show()
    return fig


def set_color(method):
    cmap = matplotlib.cm.get_cmap('tab10')
    return cmap(CMAP_I[method])


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t._ppf((1 + confidence) / 2., n - 1)
    return m, m-h, m+h
