import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
from math import ceil, floor
from pathlib import Path

ALGS = ('LSHADE', 'DM-LSHADE', '2', '3', '4', '5', '6', '7', '8', '9', '10')

cs = {'axes.spines.top': False}
sns.set_theme(style='ticks', palette='muted', rc=cs)

parser = argparse.ArgumentParser()
parser.add_argument('--merge', action='store_true')
merge = parser.parse_args().merge

for fun in [10]:
    # Diretórios
    root = Path(f'./output/logs/{fun}')
    files = [path for path in root.iterdir() if path.is_file() and path.suffix == '.log']
    Path(f'./output/graphics/{fun}').mkdir(parents=True, exist_ok=True)

    # Arquivos
    data = list()
    for file in files:
        _, var, alg = file.name.removesuffix('.log').split('-')
        with open(file) as f:
            for line in f:
                time, fit, _, d = line.strip().split(',')
                data.append({
                    'Alg': int(alg),
                    'Var': int(var),
                    'Diversity': int(d),
                    'Time': float(time),
                    'Fit': float(fit),
                })

    # DataFrame
    df = pd.DataFrame(data, columns=['Alg', 'Var', 'Diversity', 'Time', 'Fit'])

    # Gráficos
    for d in [1]:
        for var in [10, 20, 30, 50, 100]:
            base = df[(df['Diversity'] == d) & (df['Var'] == var)]
            output = f'./output/graphics/{fun}/Merge_d={d}_var={var}.png'
            times = base.groupby('Alg')['Time'].mean().reset_index()
            fits = base.groupby('Alg')['Fit'].mean().reset_index()
            if merge:
                fig, ax1 = plot.subplots(figsize=(10, 6))
                line1, = ax1.plot(fits['Alg'], fits['Fit'], 'o-', markersize=6, label='Avg. Fitness', color='royalblue', markeredgecolor='w')
                ax1.set_ylabel('Avg. Fitness', fontsize=12)
                ax1.set_xlabel('Processors', fontsize=12)
                ax1.tick_params(axis='y')

                end = max(1, ceil(fits['Fit'].max()))
                start = max(0, floor(fits['Fit'].min()))
                ticks = [start+((end-start)/5)*i for i in range(6)]
                ax1.set_ylim(start-0.25, end)
                ax1.set_yticks(ticks)

                ax2 = ax1.twinx()
                line2, = ax2.plot(times['Alg'], times['Time'], 's-', markersize=6, label='Time (s)', color='darkorange', markeredgecolor='w')
                ax2.set_ylabel('Time (s)', fontsize=12)
                ax2.tick_params(axis='y')

                end = max(1, ceil(times['Time'].max()))
                start = max(0, floor(times['Time'].min()))
                ticks = [start+((end-start)/5)*i for i in range(6)]
                ax2.set_ylim(start-0.25, end)
                ax2.set_yticks(ticks)

                ax1.legend(handles=[line1, line2], loc='upper left', fontsize=10)
                ax1.set_xticks(range(11), ALGS, fontsize=10)
                plot.tight_layout()
                plot.savefig(output)
                plot.close()
            else:
                for s in ['Fit', 'Time']:
                    plot.figure(figsize=(10, 6))
                    mean = base.groupby('Alg')[s].mean().reset_index()
                    output = f'./output/graphics/{fun}/{s}_d={d}_var={var}.png'

                    plot.plot(mean['Alg'], mean[s], 'o-', markersize=8, color='royalblue' if s == 'Fit' else 'darkorange')
                    plot.xticks(range(11), ALGS, fontsize=10)
                    plot.xlabel('Algorithms', fontsize=12)
                    plot.ylabel(f'Mean {s}', fontsize=12)
                    plot.tight_layout()
                    plot.savefig(output)
                    plot.close()
