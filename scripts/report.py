import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


if __name__ == '__main__':
	# Argumentos do CLI
	parser = argparse.ArgumentParser()
	parser.add_argument('fun', type=int, help='Número da função: 1-30')
	args = parser.parse_args()

	# Filtra arquivos
	root = Path(f'./output/logs/{args.fun}')
	files = [path for path in root.iterdir() if path.is_file() and path.suffix == '.log']

	# Leitura dos Arquivos
	data = list()
	for file in files:
		_, var, p = file.name.removesuffix('.log').split('-')
		with open(file) as f:
			for line in f:
				time, fit, _, d = line.strip().split(',')
				data.append({
					'Diversidade': int(d),
					'Var': int(var),
					'P': int(p),
					'Time': float(time),
					'Fit': float(fit),
				})

	# Criar o DataFrame
	df = pd.DataFrame(data, columns=['Diversidade', 'Var', 'P', 'Time', 'Fit'])

	# Gráficos Básicos
	for s in ['Time', 'Fit']:
		for d in [0, 1]:
			for var in [10, 20, 50, 100]:
				filtered_df = df[(df['Diversidade'] == d) & (df['Var'] == var)]
				mean = filtered_df.groupby('P')[s].mean().reset_index()

				plt.figure(figsize=(8, 6))
				plt.plot(mean['P'], mean[s], marker='o', linestyle='-', color='b')

				plt.title(f'Gráfico de Média de {s} x P (Diversidade={d}, Var={var})')
				plt.xticks([1, 2, 4, 6, 8])
				plt.ylabel(f'Média de {s}')
				plt.xlabel('Processos')

				plt.savefig(f'./output/report/img/{s}_x_P_d={d}_var={var}.png', dpi=300)
				plt.close()
