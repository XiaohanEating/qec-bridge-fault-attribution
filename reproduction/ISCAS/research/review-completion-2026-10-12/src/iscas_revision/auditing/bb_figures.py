"""Publication curves for all registered BB finite-noise points."""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from ..artifacts import WORK, read_json, commit_json, digest
from ..figures import style, export


def render(require_complete=True):
    source = WORK/'results/bb-components.json'
    data = read_json(source)
    if require_complete:
        assert data['finite_complete'] and len(data['finite_points']) == 48
    points = {(p['instance'], p['order'], p['eps']): p for p in data['finite_points']}
    rates = np.array([1e-9, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
    orders = ('forward', 'macro_reverse', 'check_reverse')
    names = ('Forward', 'Macro reverse', 'Check reverse')
    colors = ('#767676', '#B64342', '#0F4D92')
    rules = ('original', 'strong', 'single_fault_lookup')
    style()
    fig, axes = plt.subplots(2, 3, figsize=(8.1, 5.2), sharex=True, sharey=True)
    for row, instance in enumerate(('bb-72', 'bb-108')):
        for column, rule in enumerate(rules):
            ax = axes[row, column]
            for order, name, color, marker in zip(orders, names, colors, ('o', 's', '^')):
                values = [points.get((instance, order, float(eps))) for eps in rates]
                curve = np.array([p['decoder'][rule]['probability'] if p else np.nan for p in values])
                ax.plot(rates, np.where(curve > 0, curve, np.nan), color=color, marker=marker,
                    markersize=3.5, linewidth=1.1, label=name)
                for eps, point in zip(rates, values):
                    if point is None:
                        continue
                    value = point['decoder'][rule]
                    probability = value['probability']
                    lo, hi = value['marginal_interval']
                    if value['failures'] == 0:
                        ax.scatter(eps, hi, marker='v', s=22, facecolors='none', edgecolors=color,
                            linewidths=.9, zorder=4)
                    else:
                        ax.errorbar(eps, probability, yerr=[[probability-lo], [hi-probability]],
                            color=color, capsize=1.5, linewidth=.8, fmt='none', alpha=.8)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(1e-7, 1.4)
            ax.set_xlim(6e-10, 1.6e-4)
            ax.set_title(instance.upper()+' / '+('Original', 'Strong', 'Lookup')[column], fontsize=9)
            ax.set_xticks([1e-9, 1e-7, 1e-5, 1e-4])
            ax.tick_params(labelsize=8)
            if column == 0:
                ax.set_ylabel('Joint logical-Z failure', fontsize=9)
            if row == 1:
                ax.set_xlabel(r'Physical noise probability $\epsilon$', fontsize=9)
    axes[0, 0].legend(loc='lower right', fontsize=7.5, facecolor='white', edgecolor='none', framealpha=.85)
    fig.tight_layout(pad=.7, h_pad=1, w_pad=.8)
    name = 'bb-finite-noise' if data['finite_complete'] else 'bb-finite-noise-preview'
    caption = ('Joint logical-Z failure in open-grid BB bridges under all three decision rules. '
        'Each point uses 32,768 physical samples. Bars show marginal 95% exact binomial intervals; '
        'open downward triangles show the upper endpoints of those intervals when zero failures were observed. '
        'Orders have matching ideal action, directed CX counts, slots and per-site cumulative idling. '
        'Preview gaps correspond to unfinished registered points.' if not data['finite_complete'] else
        'Joint logical-Z failure in open-grid BB bridges under all three decision rules. '
        'Each point uses 32,768 physical samples. Bars show marginal 95% exact binomial intervals; '
        'open downward triangles show the upper endpoints of those intervals when zero failures were observed. '
        'Orders have matching ideal action, directed CX counts, slots and per-site cumulative idling.')
    export(fig, name, [source], caption)
    commit_json(WORK/'figures'/(name+'-validation.json'), dict(complete=data['finite_complete'],
        plotted_points=len(points), source_sha256=digest(__file__), input_sha256=digest(source),
        zero_count_representation='Upper endpoint of two-sided 95% Clopper-Pearson interval',
        scientific_inputs_modified=False))
    print(dict(complete=data['finite_complete'], plotted_points=len(points), figure=name), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    render(require_complete=not args.partial)
