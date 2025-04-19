import argparse
import pandas as pd
from pathlib import Path



if __name__ == '__main__':
	# Argumentos do CLI
	parser = argparse.ArgumentParser()
	parser.add_argument('fun', type=str, help='Número da função: 1-30')
	args = parser.parse_args()

	# Filtra arquivos
	dir = Path('./output/logs')
	files = [
		file for file in dir.iterdir()
		if file.is_file() and file.stem.startswith(args.fun) and file.suffix == '.log'
	]

	# DataFrame
	df = pd.DataFrame(columns=["Diversidade", "Var", "P", "Time", "Fit"])

	for file in files:
		_, var, p = file.name.removesuffix('.log').split('-')
		with open(file) as f:
			for line in f:
				time, fit, div  = line.strip().split(',')
				df = df.append({
					"Diversidade": div,
					"Var": var,
					"P": p,
					"Time": time,
					"Fit": fit,
				}, ignore_index=True)



	print(df)

# Gráficos
# - Time x p
# - Fitness x p
# - Fitness x Population_Size (for each p)
# - Time x Population_Size (for each p)
