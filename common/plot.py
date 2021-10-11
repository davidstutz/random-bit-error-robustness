from . import utils
from .log import log, LogLevel

if not utils.display() and not utils.notebook():
    log('[Error] DISPLAY not found, plot not available!', LogLevel.ERROR)

import matplotlib
from matplotlib import pyplot
import numpy


# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
# https://matplotlib.org/users/usetex.html
#matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
# https://matplotlib.org/users/customizing.html
matplotlib.rcParams['lines.linewidth'] = 1
#matplotlib.rcParams['figure.figsize'] = (30, 10)


# from http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
color_brewer_ = numpy.array([
    [166, 206, 227],
    [31, 120, 180],
    [251, 154, 153],
    [178, 223, 138],
    [51, 160, 44],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [245, 245, 145],
    [177, 89, 40],

], dtype=float)
color_brewer = color_brewer_/255.

marker_brewer = [
    'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
]


def label(ax=pyplot.gca(), legend=False, **kwargs):
    """
    Label axes, title etc.

    :param legend: whether to add and return a legend
    :type legend: bool
    :return: legend
    :rtype: None or matplotlib.legend.Legend
    """

    title = kwargs.get('title', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    xscale = kwargs.get('xscale', None)
    yscale = kwargs.get('yscale', None)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xscale is not None:
        if xscale == 'symlog':
            linthreshx = kwargs.get('linthreshx', 10 ** -10)
            ax.set_xscale(xscale, linthreshx=linthreshx)
        else:
            ax.set_xscale(xscale)
    if yscale is not None:
        if yscale == 'symlog':
            linthreshy = kwargs.get('linthreshy', 10 ** -10)
            ax.set_yscale(yscale, linthreshy=linthreshy)
        else:
            ax.set_yscale(yscale)

    xmax = kwargs.get('xmax', None)
    xmin = kwargs.get('xmin', None)
    ymax = kwargs.get('ymax', None)
    ymin = kwargs.get('ymin', None)

    if xmax is not None:
        ax.set_xbound(upper=xmax)
    if xmin is not None:
        ax.set_xbound(lower=xmin)
    if ymax is not None:
        ax.set_ybound(upper=ymax)
    if ymin is not None:
        ax.set_ybound(lower=ymin)

    ax.figure.set_size_inches(kwargs.get('w', 6), kwargs.get('h', 6))

    # This is fixed stuff.
    ax.grid(b=True, which='major', color=(0.5, 0.5, 0.5), linestyle='-')
    ax.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--')

    legend_loc = kwargs.get('legend_loc', 'upper left')
    legend_anchor = kwargs.get('legend_anchor', (1, 1.05))

    if legend:
        legend_ = ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor)
        legend_.get_frame().set_alpha(None)
        legend_.get_frame().set_facecolor((1, 1, 1, 0.5))
        return legend_


def errorbar(x, y, yerrl, yerrh, labels=None, ax=pyplot.gca(), **kwargs):
    """
    Error bar plot.

    :param data: vector of data to plot
    :type data: numpy.ndarray
    :param labels: optional labels
    :type labels: [str]
    """

    if isinstance(x, numpy.ndarray):
        assert len(x.shape) == 1 or len(x.shape) == 2, ' only one- or two-dimensional data can be line-plotted'
        assert len(y.shape) == 1 or len(y.shape) == 2, ' only one- or two-dimensional data can be line-plotted'
        assert len(yerrl.shape) == 1 or len(yerrl.shape) == 2, ' only one- or two-dimensional data can be line-plotted'
        assert len(yerrh.shape) == 1 or len(yerrh.shape) == 2, ' only one- or two-dimensional data can be line-plotted'

        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(y.shape) == 1:
            y = y.reshape((1, -1))
        if len(yerrl.shape) == 1:
            yerrl = yerrl.reshape((1, -1))
        if len(yerrh.shape) == 1:
            yerrh = yerrh.reshape((1, -1))

        num_labels = x.shape[0]
    elif isinstance(x, list):
        assert isinstance(y, list)
        assert isinstance(yerrl, list)
        assert isinstance(yerrh, list)

        assert len(x) == len(y)
        assert len(x) == len(yerrl)
        assert len(x) == len(yerrh)

        for i in range(len(x)):
            assert x[i].shape[0] == y[i].shape[0]
            assert x[i].shape[0] == yerrl[i].shape[0]
            assert x[i].shape[0] == yerrh[i].shape[0]

        num_labels = len(x)
    else:
        assert False

    has_labels = (labels is not None)
    if not has_labels:
        labels = [None] * num_labels
    assert len(labels) <= len(color_brewer), 'currently a maxmimum of %d different labels are supported' % len(color_brewer)

    for i in range(num_labels):
        ax.errorbar(x[i], y[i], yerr=(yerrl[i], yerrh[i]), fmt=kwargs.get('fmt', ''), markeredgewidth=kwargs.get('markeredgewidth', 2.5), capsize=kwargs.get('capsize', 5),
                                   color=tuple(color_brewer[i]), label=labels[i], marker=marker_brewer[i], linewidth=kwargs.get('linewidth', 1))

    ax.legend()
    label(ax, legend=has_labels, **kwargs)
