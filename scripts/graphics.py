import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from pathlib import Path

font1 = FontProperties(weight='bold', size=20)
font2 = FontProperties(weight='bold', size=15)
ALGS = ('LDE', 'DM', '2', '3', '4', '5', '6', '7', '8', '9', '10')
sns.set_theme(style='ticks', palette='muted', rc={'axes.spines.top': False})

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
			if merge:
				fits = base.groupby('Alg')['Fit'].mean().reset_index()
				times = base.groupby('Alg')['Time'].mean().reset_index()
				output = f'./output/graphics/{fun}/Merge_d={d}_var={var}.pdf'
				fig, ax1 = plt.subplots(figsize=(10, 6)); ax2 = ax1.twinx()

				line3 = ax1.axvline(x=8, color='red', linestyle='--', linewidth=1.5)
				line1, = ax1.plot(fits['Alg'], fits['Fit'], 'o-', markersize=10, label='Avg. Fitness', color= 'royalblue', markeredgecolor='w', linewidth=3)
				line2, = ax2.plot(times['Alg'], times['Time'], 's-', markersize=10, label='Time (s)', color='darkorange', markeredgecolor='w', linewidth=3)
				
				ax1.set_xlabel('Processors', fontproperties=font1)
				ax1.set_xticks(range(11), ALGS, fontproperties=font2)
				ax1.set_ylabel('Avg. Fitness', fontproperties=font1)
				ax2.set_ylabel('Time (s)', fontproperties=font1)

				for label in ax1.get_yticklabels():
					label.set_fontproperties(font2)
				for label in ax2.get_yticklabels():
					label.set_fontproperties(font2)

				ax1.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=6))

				ax1.legend(
					handles=[line1, line2, line3],
					labels=['Avg. Fitness', 'Time (s)', 'Performance Transition'],
					loc='upper center',
					bbox_to_anchor=(0.5, 1.1),
					ncol=3,
					fontsize=12,
					frameon=True,
					prop=font2
				)

				plt.tight_layout()
				plt.savefig(output, bbox_inches='tight')
				plt.close()
			else:
				for s in ['Fit', 'Time']:
					plt.figure(figsize=(10, 6))
					mean = base.groupby('Alg')[s].mean().reset_index()
					color = 'royalblue' if s == 'Fit' else 'darkorange'
					label = 'Avg. Fitness' if f == 'Fit' else 'Time (s)'
					output = f'./output/graphics/{fun}/{s}_d={d}_var={var}.pdf'
					plt.plot(mean['Alg'], mean[s], 'o-', markersize=8, color=color)
					plt.xticks(range(11), ALGS, fontsize=15)
					plt.xlabel('Processors', fontsize=15)
					plt.ylabel(label, fontsize=15)
					plt.tight_layout()
					plt.savefig(output)
					plt.close()
