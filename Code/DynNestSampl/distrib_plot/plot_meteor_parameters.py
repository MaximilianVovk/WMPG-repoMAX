#!/usr/bin/env python3
r"""Recursively plot posterior summaries from *_results_table*.tex.

Requirements: Python 3.8+, numpy, matplotlib.
Examples (PowerShell):
    python .\plot_meteor_parameters.py .
    python .\plot_meteor_parameters.py "C:\path\EMCCD" "C:\path\CAMO"
    python .\plot_meteor_parameters.py . --linear --individual

A directory component named CAMO (case-insensitive), anywhere above a file,
marks that entry as CAMO. Non-CAMO entries come first; CAMO entries go last.
Three figures are saved, each with meteor IDs on the x-axis and CAMO last:
1. all_parameter_ranges: mode, mean, median, best simulation, 95% bounds.
2. all_parameter_ranges_square: connected medians, blue/orange error bars,
   best simulation, and a grey pyplot.boxplot of the fitted medians.
3. tauH_modelling_variants_comparison_fixed: six selected parameters:
   mass, speed, density, erosion coefficient, sqrt(ml*mu), rho**(2/3)/eta.
Derived centres are functions of marginal medians, not posterior medians.
Derived ranges propagate marginal endpoints and are not joint 95% intervals.
Grey boxes describe between-fit spread, not a combined posterior. CAMO is
included; distinct fits/stages count separately. No synthetic posteriors.
Source files are never modified. Re-running overwrites matching output files.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# key: (plain-language title, preferred logarithmic scale)
PARAMETERS = {
    'v0': ('Initial speed', False),
    'm0': ('Initial mass', True),
    'rho': ('Bulk density', False),
    'sigma': ('Ablation coefficient', False),
    'he': ('Erosion onset height', False),
    'eta': ('Erosion coefficient', True),
    's': ('Mass distribution index', False),
    'ml': ('Lower grain-mass limit', True),
    'mu': ('Upper grain-mass limit', True),
    'he2': ('Second erosion onset height', False),
    'eta2': ('Second erosion coefficient', True),
    'rho2': ('Second-stage density', False),
    'sigmalag': ('Lag noise parameter', False),
    'sigmalum': ('Luminosity noise parameter', True),
}
STATS = ('lower', 'best_fit', 'mode', 'mean', 'median', 'upper')
STYLES = (
    ('median', 'o', '#1672a0', -.15),
    ('mean', 'D', '#e08723', -.05),
    ('mode', '^', '#9a4a96', .05),
    ('best_fit', 'x', '#303a42', .15),
)


def clean_key(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


def number(text):
    """Read plain decimal/e-notation or LaTeX mantissa times 10^{exponent}."""
    text = text.strip().strip('$').replace('−', '-').replace('–', '-')
    text = re.sub(r'\\(?:,|;|!|quad|times|cdot)', '', text)
    text = re.sub(r'\s+', '', text)
    power = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))10\^\{?([+-]?\d+)\}?', text)
    if power:
        value = float(power[1]) * 10.0 ** int(power[2])
    else:
        value = float(text)
    if not math.isfinite(value):
        raise ValueError('non-finite value: ' + text)
    return value


def parse_table(path):
    text = path.read_text(encoding='utf-8-sig')
    columns = None
    parameters = {}
    aliases = {'25ci': 'lower', '975ci': 'upper', 'bestfit': 'best_fit',
               'mode': 'mode', 'mean': 'mean', 'median': 'median'}
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = re.split(r'(?<!\\)%', raw, maxsplit=1)[0].strip()
        if '&' not in line:
            continue
        cells = [c.strip() for c in line.split('\\\\', 1)[0].split('&')]
        if clean_key(cells[0]) == 'parameter':
            columns = [aliases.get(clean_key(c)) for c in cells[1:]]
            if len(columns) != 6 or set(columns) != set(STATS):
                raise ValueError('unrecognized summary column headers')
            continue
        if not line.startswith('$'):
            continue
        if columns is None:
            raise ValueError('parameter row precedes recognized header')
        if len(cells) != len(columns) + 1:
            raise ValueError('wrong number of columns on line {}'.format(line_no))
        match = re.match(r'\$([^$]+)\$', cells[0])
        if not match:
            raise ValueError('unrecognized parameter on line {}'.format(line_no))
        key = clean_key(match[1])
        if key in parameters:
            raise ValueError('duplicate parameter ' + key)
        values = dict(zip(columns, map(number, cells[1:])))
        if not values['lower'] <= values['median'] <= values['upper']:
            raise ValueError('bounds/median are inconsistent for ' + key)
        parameters[key] = dict(label=cells[0], **values)
    if not parameters:
        raise ValueError('no posterior rows found')
    event = re.search(r'\d{8}_\d{6}', path.name)
    if event is None:
        event = re.search(r'\d{8}_\d{6}', str(path.parent))
    event = event.group() if event else path.stem.split('_results_table')[0]
    return dict(event=event, camo=any(p.name.casefold() == 'camo' for p in path.parents),
                source=str(path), parameters=parameters)


def plot_panel(ax, key, entries, linear=False):
    available = [(i, e['parameters'][key]) for i, e in enumerate(entries)
                 if key in e['parameters']]
    x = np.array([i for i, _ in available])
    rows = [r for _, r in available]
    lower = np.array([r['lower'] for r in rows])
    upper = np.array([r['upper'] for r in rows])
    ax.vlines(x, lower, upper, color='#8996a2', linewidth=1.8, zorder=2)
    ax.hlines(lower, x - .21, x + .21, color='#8996a2', linewidth=1.4)
    ax.hlines(upper, x - .21, x + .21, color='#8996a2', linewidth=1.4)
    for stat, marker, color, offset in STYLES:
        ax.scatter(x + offset, [r[stat] for r in rows], marker=marker,
                   color=color, s=32, linewidths=1.3, zorder=3)
    title, preferred_log = PARAMETERS.get(key, (key, False))
    use_log = not linear and preferred_log and all(r[s] > 0 for r in rows for s in STATS)
    if use_log:
        ax.set_yscale('log')
    ax.set_title(title + (' · log scale' if use_log else ''), loc='left', pad=12)
    ax.set_ylabel(rows[0]['label'])
    labels = [e['display'].replace('_', '\n').replace(' CAMO', '\nCAMO') for e in entries]
    ax.set_xticks(np.arange(len(entries)))
    ax.set_xticklabels(labels, fontsize=8, rotation=0 if len(entries) <= 12 else 45,
                       ha='center' if len(entries) <= 12 else 'right')
    ax.set_xlim(-.55, len(entries) - .45)
    non_camo = sum(not e['camo'] for e in entries)
    if 0 < non_camo < len(entries):
        ax.axvline(non_camo - .5, color='#aab3ba', linestyle='--', linewidth=1)
    if len(rows) != len(entries):
        ax.text(.02, .97, 'Blank positions: parameter not reported', transform=ax.transAxes,
                va='top', fontsize=9, color='#59636b')
    ax.grid(axis='y', which='major', alpha=.23)
    ax.set_axisbelow(True)
    ax.margins(y=.17)



def derived_row(parameters, kind, stage=1):
    """Monotonic endpoint propagation, without assuming posterior independence.

    The rectangle of marginal intervals is NOT a joint 95% credible region.
    Evaluating f at marginal medians does NOT give a posterior median of f.
    """
    if kind == 'grain':
        suffix = '' if stage == 1 else '2'
        left, right = parameters.get('ml' + suffix), parameters.get('mu' + suffix)
        if left is None or right is None:
            return None
        if min(left[k] for k in ('lower', 'upper', 'median', 'best_fit')) <= 0 or min(
                right[k] for k in ('lower', 'upper', 'median', 'best_fit')) <= 0:
            return None
        return {stat: math.sqrt(left[stat] * right[stat])
                for stat in ('lower', 'upper', 'median', 'best_fit')}
    eta = parameters.get('eta' if stage == 1 else 'eta2')
    # With no fitted second density, the model's single density is reused.
    rho = parameters.get('rho2', parameters.get('rho')) if stage == 2 else parameters.get('rho')
    if eta is None or rho is None:
        return None
    if min(eta[k] for k in ('lower', 'upper', 'median', 'best_fit')) <= 0 or min(
            rho[k] for k in ('lower', 'upper', 'median', 'best_fit')) <= 0:
        return None
    return dict(lower=rho['lower'] ** (2/3) / eta['upper'],
                upper=rho['upper'] ** (2/3) / eta['lower'],
                median=rho['median'] ** (2/3) / eta['median'],
                best_fit=rho['best_fit'] ** (2/3) / eta['best_fit'])


def selected_rows(entry, key):
    parameters = entry['parameters']
    if key in ('grain', 'pi'):
        return [(stage, derived_row(parameters, key, stage)) for stage in (1, 2)]
    if key == 'eta':
        return [(1, parameters.get('eta')), (2, parameters.get('eta2'))]
    return [(1, parameters.get(key))]


def comparison_plot(entries, keys, output, args, warnings, selected=False):
    """Connected meteor estimates, error bars, and a final grey summary box."""
    specs = [
        ('m0', 'Initial mass', r'$m_0$ [kg]', True),
        ('v0', 'Initial speed', r'$v_0$ [km/s]', False),
        ('rho', 'Bulk density', r'$\rho_b$ [kg/m$^3$]', False),
        ('eta', 'Erosion coefficient', r'$\eta$ [kg/MJ]', PARAMETERS['eta'][1]),
        ('grain', 'Geometric-mean grain mass', r'$\sqrt{m_l m_u}$ [kg]', True),
        ('pi', 'Erosion parameter', r'$\Pi_{er}=\rho_b^{2/3}/\eta$ [MJ kg$^{-1/3}$ m$^{-2}$]', True),
    ] if selected else [
        (key, PARAMETERS.get(key, (key, False))[0],
         next(e['parameters'][key]['label'] for e in entries if key in e['parameters']),
         PARAMETERS.get(key, (key, False))[1]) for key in keys]
    cols = min(3, len(specs))
    nrows = math.ceil(len(specs) / cols)
    width = max(8.5, .7 * (len(entries) + 2))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * width, nrows * 5.2 + 2), squeeze=False)
    x = np.arange(len(entries))
    labels = [e['display'].replace('_', '\n').replace(' CAMO', '\nCAMO') for e in entries]
    summary_x = len(entries) + .5
    split = sum(not e['camo'] for e in entries)
    derived_export = []
    for ax, (key, title, label, prefer_log) in zip(axes.flat, specs):
        estimates = {stage: np.full(len(entries), np.nan) for stage in (1, 2)}
        summary_values, extent = [], []
        stage2 = selected and key in ('eta', 'grain', 'pi')
        for i, entry in enumerate(entries):
            series = selected_rows(entry, key) if selected else [(1, entry['parameters'].get(key))]
            for stage, row in series:
                if row is None:
                    continue
                centre, lo, hi = row['median'], row['lower'], row['upper']
                estimates[stage][i] = centre
                color = '#dc8425' if entry['camo'] else '#1672a0'
                dx = (-.07 if stage == 1 else .07) if stage2 else 0
                ax.errorbar(i + dx, centre, yerr=[[centre - lo], [hi - centre]],
                            fmt='o' if stage == 1 else '^', color=color, ecolor=color,
                            markersize=4, elinewidth=1.2, capsize=3, zorder=4)
                ax.scatter(i + dx, row['best_fit'], marker='x', color='#303a42', s=25, zorder=5)
                summary_values.append(centre)
                extent.extend([lo, hi, row['best_fit']])
                if selected and key in ('grain', 'pi'):
                    derived_export.append(dict(event=entry['display'], parameter=key, stage=stage,
                                               centre_from_marginal_medians=centre,
                                               propagated_lower=lo, propagated_upper=hi,
                                               best_simulation=row['best_fit']))
        for stage in (1, 2):
            dx = (-.07 if stage == 1 else .07) if stage2 else 0
            linestyle = '-' if stage == 1 else '--'
            for start, stop, color in ((0, split, '#1672a0'), (split, len(entries), '#dc8425')):
                ax.plot(x[start:stop] + dx, estimates[stage][start:stop], color=color,
                        linestyle=linestyle, linewidth=1.3, zorder=3)
        use_log = prefer_log and not args.linear and extent and min(extent) > 0
        if use_log:
            ax.set_yscale('log')
        ax.set_title(title + (' · log scale' if use_log else ''), fontweight='normal', pad=12)
        ax.set_ylabel(label, fontweight='normal')
        plt.sca(ax)
        if summary_values:
            plt.boxplot([summary_values], positions=[summary_x], widths=.55,
                        whis=(2.5, 97.5), patch_artist=True, manage_ticks=False,
                        boxprops=dict(facecolor='#bdbdbd', edgecolor='#777777', alpha=.8),
                        medianprops=dict(color='#333333', linewidth=1.8),
                        whiskerprops=dict(color='#777777', linewidth=1.2),
                        capprops=dict(color='#777777', linewidth=1.2),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor='#888888',
                                        markeredgecolor='none', alpha=.65))
        ax.axvline(len(entries) - .25, color='#aaaaaa', linestyle=':', linewidth=1)
        ax.set_xticks(list(x) + [summary_x])
        summary_label = ('All fits/stages' if stage2 else 'All fits') + '\n(n={})'.format(len(summary_values))
        ax.set_xticklabels(labels + [summary_label], fontsize=8, rotation=0 if len(entries) <= 12 else 45,
                           ha='center' if len(entries) <= 12 else 'right')
        ax.set_xlim(-.55, summary_x + .6)
        if 0 < split < len(entries):
            ax.axvline(split - .5, color='#aaaaaa', linestyle='--', linewidth=1)
        if not summary_values:
            ax.text(.5, .5, 'Required parameters not reported', transform=ax.transAxes, ha='center')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.16)
    for ax in list(axes.flat)[len(specs):]:
        ax.set_visible(False)
    title = 'Selected meteor parameters from dynamic nested sampling' if selected else 'Meteor parameters from dynamic nested sampling'
    fig.suptitle(title, fontsize=21, fontweight='normal', y=.99)
    handles = [Line2D([0], [0], color='#1672a0', marker='o', label='Median / derived estimate'),
               Line2D([0], [0], color='#dc8425', marker='o', label='CAMO'),
               Line2D([0], [0], color='#303a42', marker='x', linestyle='', label='Best simulation'),
               Patch(facecolor='#bdbdbd', edgecolor='#777777', label='Distribution of fitted estimates')]
    if selected:
        handles.append(Line2D([0], [0], color='#777777', marker='^', linestyle='--', label='Second erosion stage'))
    else:
        handles[0].set_label('Median')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .957), ncol=len(handles), frameon=False)
    # note = ('Error bars: reported 2.5–97.5% bounds. Grey boxes: quartiles of fitted estimates; percentile whiskers (2.5–97.5%).\n'
    #         'Grey summary includes CAMO; distinct fits/stages count separately. It is not a combined posterior.')
    # if selected:
    #     note += ('\nDerived centres use marginal medians; their bars propagate marginal endpoints, not joint 95% credible intervals.'
    #              '\nSecond-stage grain mass requires explicit second-stage mass limits; second-stage Pi uses rho2 when reported, otherwise rho.')
    # fig.text(.5, .012, note, ha='center', fontsize=10, fontweight='normal')
    fig.tight_layout(rect=(0, .12 if selected else .065, 1, .92), h_pad=2.4, w_pad=2)
    basename = 'modelling_variants_comparison_fixed' if selected else 'all_parameter_ranges_square'
    for extension in ('png', 'svg'):
        fig.savefig(output / (basename + '.' + extension), dpi=args.dpi)
    plt.close(fig)
    if selected:
        (output / 'derived_parameter_estimates.json').write_text(json.dumps(derived_export, indent=2), encoding='utf-8')
        warnings.append('Derived grain mass and Pi: functions of marginal medians; endpoint ranges are not joint 95% credible intervals.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('roots', nargs='*', default=[r'C:\Users\maxiv\Documents\UWO\Papers\0.6)Tah_erc\Results-dynesty\EMCCD\EMCCD-cz_good'], help='One or more folders to search recursively')
    parser.add_argument('--output', type=Path, help='Default: meteor_parameter_plots under first search folder')
    parser.add_argument('--individual', action='store_true', help='Also save a separate figure per parameter')
    parser.add_argument('--linear', action='store_true', help='Use linear y-axes for every parameter')
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error('--dpi must be positive')
    roots = [Path(p).expanduser().resolve() for p in args.roots]
    for root in roots:
        if not root.is_dir():
            parser.error('Folder does not exist: ' + str(root))
    output = (args.output or roots[0] / 'meteor_parameter_plots').expanduser().resolve()
    # A set avoids reading a file twice when search roots overlap.
    files = sorted({p.resolve() for root in roots for p in root.rglob('*')
                    if p.is_file() and re.search(r'_results_table(?:\(\d+\))?\.tex$', p.name, re.I)})
    entries, warnings = [], []
    for path in files:
        try:
            entries.append(parse_table(path))
        except (ValueError, OSError, UnicodeError) as exc:
            warnings.append('SKIPPED {}: {}'.format(path, exc))
    if not entries:
        for warning in warnings:
            print(warning)
        parser.error('No readable *_results_table.tex files found below the supplied folders.')
    entries.sort(key=lambda e: (e['camo'], e['event'], e['source']))
    # Keep distinct fits of the same meteor, including CAMO versus non-CAMO.
    totals = Counter((e['event'], e['camo']) for e in entries)
    seen = Counter()
    for e in entries:
        identity = (e['event'], e['camo'])
        seen[identity] += 1
        e['display'] = e['event'] + (' CAMO' if e['camo'] else '')
        if totals[identity] > 1:
            e['display'] += ' [{}]'.format(seen[identity])
            warnings.append('Duplicate label retained as {}: {}'.format(e['display'], e['source']))
        for key, row in e['parameters'].items():
            for stat in ('mode', 'mean', 'best_fit'):
                if not row['lower'] <= row[stat] <= row['upper']:
                    warnings.append('{} {}: {}={} is outside reported bounds; preserved.'.format(
                        e['display'], key, stat, row[stat]))
    keys_found = set(k for e in entries for k in e['parameters'])
    keys = [k for k in PARAMETERS if k in keys_found] + sorted(keys_found - set(PARAMETERS))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'parameter_values.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    (output / 'plot_notes.txt').write_text('\n'.join(warnings) or 'No warnings.', encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
                         'font.weight': 'normal', 'axes.titleweight': 'normal', 'axes.labelweight': 'normal'})
    legend = [Line2D([0], [0], color='#8996a2', marker='_', label='2.5–97.5% interval')]
    legend += [Line2D([0], [0], linestyle='', marker=m, color=c,
                      label='Best simulation' if s == 'best_fit' else s.title()) for s, m, c, _ in STYLES]
    ncols = min(2, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    panel_width = max(8.5, .58 * len(entries))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_width, 4.8 * nrows + 1.4), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        plot_panel(ax, key, entries, args.linear)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    fig.suptitle('Meteor physical parameters and posterior ranges', fontsize=21, y=.99)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .968), ncol=5, frameon=False)
    fig.text(.5, .012, '{} entries | CAMO entries on the right | Markers offset for readability.\n'
             'Reported bounds and statistics preserved; missing parameters left blank.'.format(len(entries)),
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .935), h_pad=2.4, w_pad=2)
    for ext in ('png', 'svg'):
        fig.savefig(output / ('all_parameter_ranges.' + ext), dpi=args.dpi)
    plt.close(fig)
    comparison_plot(entries, keys, output, args, warnings)
    comparison_plot(entries, keys, output, args, warnings, selected=True)
    (output / 'plot_notes.txt').write_text('\n'.join(warnings) or 'No warnings.', encoding='utf-8')
    if args.individual:
        for key in keys:
            fig, ax = plt.subplots(figsize=(max(12, .7 * len(entries)), 7))
            plot_panel(ax, key, entries, args.linear)
            fig.legend(handles=legend, loc='upper center', ncol=5, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, .9))
            for ext in ('png', 'svg'):
                fig.savefig(output / (key + '_ranges.' + ext), dpi=args.dpi)
            plt.close(fig)
    print('Plotted {} entries ({} CAMO), {} parameters.'.format(
        len(entries), sum(e['camo'] for e in entries), len(keys)))
    for e in entries:
        print('  {} <- {}'.format(e['display'], e['source']))
    if warnings:
        print('{} notes/warnings: see plot_notes.txt'.format(len(warnings)))
    print('Output: {}'.format(output))


if __name__ == '__main__':
    main()
