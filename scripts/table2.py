import os
import warnings
import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('v', type=str)
parser.add_argument('d', type=str)
parser.add_argument('alg1', type=str)
parser.add_argument('alg2', type=str)
args = parser.parse_args()

# Constantes
ALGS = [
	"LSHADE",
	"DM-LSHADE",
	"CDM-LSHADE (n=2)",
	"CDM-LSHADE (n=3)",
	"CDM-LSHADE (n=4)",
	"CDM-LSHADE (n=5)",
	"CDM-LSHADE (n=6)",
	"CDM-LSHADE (n=7)",
	"CDM-LSHADE (n=8)",
	"CDM-LSHADE (n=9)",
	"CDM-LSHADE (n=10)",
]

HEADER = [
	"\\begin{table}\n",
	"\\begin{adjustbox}{width=\\textwidth}\n",
	"\\begin{tabular}{c r r r r l r r r r c}\n",
	"\\toprule\n",
	"\\textbf{} & \\multicolumn{4}{c}{\\textbf{"+ALGS[int(args.alg1)]+"}} ",
	"& \\multicolumn{1}{c}{} & \\multicolumn{4}{c}{\\textbf{"+ALGS[int(args.alg2)]+"}} ",
	"& \\multicolumn{1}{c}{} \\\\ \\cline{2-5} \\cline{7-10}\n",
	"\\textbf{function} & \\multicolumn{1}{c}{Best Cost} ",
	"& \\multicolumn{1}{c}{Avg. Cost} & \\multicolumn{1}{c}{Std. Cost} ",
	"& \\multicolumn{1}{c}{Avg. Time(s)} &  ",
	"& \\multicolumn{1}{c}{Best Cost} & \\multicolumn{1}{c}{Avg. Cost} ",
	"& \\multicolumn{1}{c}{Std. Cost} & \\multicolumn{1}{c}{Avg. Time(s)} ",
	"& \\multicolumn{1}{c}{\\thead{Time\\\\Gap(\\%)}} \\\\ \\toprule\n"
]

FOOTER = '''
\\end{tabular}
\\end{adjustbox}
\\caption{Legendas}
\\end{table}
'''

# Classe
class Table:
	def __init__(self, output):
		if os.path.exists(output):
			os.remove(output)
		self.output = output
		self.wsa1 = 0
		self.wsa2 = 0
		self.mtg = 0
		self.alg1 = {
			'mean_time': [0., 0],
			'mean_fit': [0., 0],
			'best_fit': [0., 0],
			'std_fit': [0., 0]
		}
		self.alg2 = {
			'mean_time': [0., 0],
			'mean_fit': [0., 0],
			'best_fit': [0., 0],
			'std_fit': [0., 0]
		}

	def update(self, fun: int, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
		self.fun = fun

		# Atualização da Significância
		fits1, fits2 = df1['Fit'], df2['Fit']
		minor = min(fits1.size, fits2.size)
		fits1 = fits1.iloc[:minor]
		fits2 = fits2.iloc[:minor]
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			_, p_valor = wilcoxon(fits1, fits2)
		self.significance = p_valor < 0.05

		# Atualização do Dados
		for alg, df in [(self.alg1, df1), (self.alg2, df2)]:
			alg['mean_time'][0] = df['Time'].mean()
			alg['mean_fit'][0]  = df['Fit'].mean()
			alg['best_fit'][0]  = df['Fit'].min()
			alg['std_fit'][0]   = df['Fit'].std()

		# Atualização das Vitórias
		for key in ['mean_time', 'std_fit', 'mean_fit', 'best_fit']:
			self.alg1[key][1] += int(self.alg1[key][0] < self.alg2[key][0])
			self.alg2[key][1] += int(self.alg2[key][0] < self.alg1[key][0])
		self.mtg += self.__timeGap()

		# Escrita na Tabela
		with open(self.output, 'a+') as latex:
			if fun ==  1: latex.write(self.__header())
			latex.write(self.__row())
			if fun == 30: latex.write(self.__footer())

	def __textbf(self, n1: float, n2: float) -> tuple:
		strf1 = '\\textbf{'+f'{n1:.3e}'+'}' if n1 < n2 else f'{n1:.3e}'
		strf2 = '\\textbf{'+f'{n2:.3e}'+'}' if n2 < n1 else f'{n2:.3e}'
		return strf1, strf2

	def __timeGap(self) -> float:
		t1 = self.alg1['mean_time'][0]
		t2 = self.alg2['mean_time'][0]
		mn, mx = min(t1, t2), max(t1, t2)
		gap = 100 * (1 - mn / mx)
		if t2 < t1: gap = -gap
		return gap

	def __header(self) -> str:
		return str().join(HEADER)

	def __row(self) -> str:
		bf1 = self.alg1['best_fit'][0]
		bf2 = self.alg2['best_fit'][0]
		mf1 = self.alg1['mean_fit'][0]
		mf2 = self.alg2['mean_fit'][0]

		strbf1, strbf2 = self.__textbf(bf1, bf2)
		strmf1, strmf2 = self.__textbf(mf1, mf2)
		if self.significance:
			self.wsa1 += int(mf1 < mf2); self.wsa2 += int(mf2 < mf1)
			strmf1, strmf2 = (strmf1+'*', strmf2) if mf1 < mf2 else (strmf1, strmf2+'*')

		row = str(self.fun)
		row += f' & {strbf1} & {strmf1}'
		for key in ['std_fit', 'mean_time']:
			row += f' & {self.alg1[key][0]:.3e}'
		row += f' & & {strbf2} & {strmf2}'
		for key in ['std_fit', 'mean_time']:
			row += f' & {self.alg2[key][0]:.3e}'
		row += f' & {self.__timeGap():.2f} \\\\\n'
		return row

	def __footer(self) -> str:
		footer = str()
		for key in ['best_fit', 'mean_fit', 'std_fit', 'mean_time']:
			footer += f' & {self.alg1[key][1]}'
			if key == 'mean_fit': footer += f' ({self.wsa1})'
		footer += ' &'
		for key in ['best_fit', 'mean_fit', 'std_fit', 'mean_time']:
			footer += f' & {self.alg2[key][1]}'
			if key == 'mean_fit': footer += f' ({self.wsa2})'
		footer += f' & {self.mtg/30:.2f} \\\\\\bottomrule{FOOTER}'
		return footer.lstrip()

# Main
if __name__ == '__main__':
	# Tabela
	os.makedirs('./output/tables/', exist_ok=True)
	table = Table(f'./output/tables/table_{args.alg1}x{args.alg2}.tex')

	# Loop das Funções
	for fun in range(1, 31):
		# Diretórios
		root = Path(f'./output/logs/{fun}')
		datas = {args.alg1: list(), args.alg2: list()}
		files = [path for path in root.iterdir() if path.is_file() and path.suffix == '.log']

		# Arquivos
		for file in files:
			_, v, p = file.name.removesuffix('.log').split('-')
			if v == args.v and p in [args.alg1, args.alg2]:
				with open(file) as log:
					for line in log:
						time, fit, _, d = line.strip().split(',')
						if d == args.d:
							datas[p].append({
								'Time': float(time),
								'Fit': float(fit),
							})

		# DataFrames
		df1 = pd.DataFrame(datas[args.alg1], columns=['Time', 'Fit'])
		df2 = pd.DataFrame(datas[args.alg2], columns=['Time', 'Fit'])
		
		# Atualização
		table.update(fun, df1, df2)
