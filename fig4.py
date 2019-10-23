from graphs.my_graph import graphs # my custom dataviz environment
import numpy as np
from data_analysis.signal_library.classical_functions import gaussian, gaussian_cumproba
mg = graphs('manuscript') # initialize a visualization env optimize for notebook display
x = np.linspace(-3,3)
x2 = np.linspace(0, 10, len(x))
fig, ax = mg.figure(figsize=(1.4,1.3), bottom=1.5, top=3., right=3.)
mg.plot(x2, 100.*gaussian(x, std=0.8)/np.max(gaussian(x, std=0.8)),
        ax=ax, no_set=True, color=mg.purple, lw=2)
mg.annotate(ax, 'complex\ndiscrimination\ntask', (.3,.85),
            color=mg.purple, ha='center', bold=True)
mg.plot(x2, 100.*gaussian_cumproba(x),
        ax=ax, no_set=True, color=mg.cyan, lw=2)
mg.set_plot(ax, xlabel='afferent modulation (Hz)', ylabel='task accuracy (%)')
ax.annotate("", xy=(0.95, -.57), xytext=(0.05, -.57),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle="->"))
mg.annotate(ax, 'arousal level', (0.5, -0.63), ha='center', va='top')
mg.annotate(ax, 'simple\ndetection\ntask', (.95,.47),
            color=mg.cyan, ha='center', bold=True)
fig.savefig('figures/Fig4.png')
