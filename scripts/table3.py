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
parser.add_argument('alg3', type=str)
args = parser.parse_args()

# Constantes
ALGS = [
	'LSHADE',
	'DM-LSHADE',
	'CDM-LSHADE (2 Processors)',
	'CDM-LSHADE (3 Processors)',
	'CDM-LSHADE (4 Processors)',
	'CDM-LSHADE (5 Processors)',
	'CDM-LSHADE (6 Processors)',
	'CDM-LSHADE (7 Processors)',
	'CDM-LSHADE (8 Processors)',
	'CDM-LSHADE (9 Processors)',
	'CDM-LSHADE (10 Processors)'
]

HEADER = [
	f'% v={args.v}, d={args.d}, alg1={args.alg1}, alg2={args.alg2}, alg3={args.alg3}\n',
	'\\begin{table}[!t]\n',
	'\t\\begin{adjustbox}{width=\\textwidth}\n',
	'\t\\begin{tabular}{c c c c c l c c c c l c c c c l c c}\n',
	'\t\t\\toprule\n',
	'\t\t\\multirow{3}{*}{\\textbf{Function}}',
	'& \\multicolumn{4}{c}{\\textbf{'+ALGS[int(args.alg1)]+'}} ',
	'& & \\multicolumn{4}{c}{\\textbf{'+ALGS[int(args.alg2)]+'}} ',
	'& & \\multicolumn{4}{c}{\\textbf{'+ALGS[int(args.alg3)]+'}} ',
	'& & \\multicolumn{2}{c}{\\textbf{\\textit{Time Comparison}}} \\\\\n',
	'\t\t\\cline{2-5} \\cline{7-10} \\cline{12-15} \\cline{17-18}\n',
	'\t\t& Best Cost & Avg. Cost & Std. Cost & Avg. Time(s) &\n',
	'\t\t& Best Cost & Avg. Cost & Std. Cost & Avg. Time(s) &\n',
	'\t\t& Best Cost & Avg. Cost & Std. Cost & Avg. Time(s) &\n',
	'\t\t& \\thead{\\footnotesize\\textit{'+ALGS[int(args.alg1)]+'}\\\\ \\footnotesize Time Gap(\\%)} ',
    '& \\thead{\\footnotesize\\textit{'+ALGS[int(args.alg2)]+'}\\\\ \\footnotesize Time Gap(\\%)} \\\\\n',
	'\t\t\\toprule\n'
]

FOOTER = '''
\t\t\\\\\\bottomrule
\t\\end{tabular}
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
		self.ws1x3 = 0
		self.ws3x1 = 0
		self.ws2x3 = 0
		self.ws3x2 = 0
		self.mtg1 = 0
		self.mtg2 = 0
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
		self.alg3 = {
			'mean_time': [0., 0],
			'mean_fit': [0., 0],
			'best_fit': [0., 0],
			'std_fit': [0., 0]
		}

	def update(self, fun: int, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> None:
		self.fun = fun

		# Atualização da Significância
		fits1, fits2, fits3 = df1['Fit'], df2['Fit'], df3['Fit']
		minor = min(fits1.size, fits2.size, fits3.size)
		fits1 = fits1.iloc[:minor]
		fits2 = fits2.iloc[:minor]
		fits3 = fits3.iloc[:minor]
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			_, p_valor1 = wilcoxon(fits1, fits3)
		self.sig1 = p_valor1 < 0.05
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			_, p_valor2 = wilcoxon(fits2, fits3)
		self.sig2 = p_valor2 < 0.05

		# Atualização do Dados
		for alg, df in [(self.alg1, df1), (self.alg2, df2), (self.alg3, df3)]:
			alg['mean_time'][0] = df['Time'].mean()
			alg['mean_fit'][0] = df['Fit'].mean()
			alg['best_fit'][0] = df['Fit'].min()
			alg['std_fit'][0] = df['Fit'].std()

		# Atualização das Vitórias
		for key in ['mean_time', 'std_fit', 'mean_fit', 'best_fit']:
			a1 = self.alg1[key][0]
			a2 = self.alg2[key][0]
			a3 = self.alg3[key][0]
			self.alg1[key][1] += int(a1 < min(a2, a3))
			self.alg2[key][1] += int(a2 < min(a1, a3))
			self.alg3[key][1] += int(a3 < min(a1, a2))
			if key == 'mean_time':
				self.mtg1 += self.__timeGap(a1, a3)
				self.mtg2 += self.__timeGap(a2, a3)

		# Escrita na Tabela
		with open(self.output, 'a+') as latex:
			if fun ==  1: latex.write(self.__header())
			latex.write(self.__row())
			if fun == 30: latex.write(self.__footer())

	def __textbf(self, n1: float, n2: float, n3: float) -> tuple:
		strf1 = '\\textbf{'+f'{n1:.3e}'+'}' if n1 < min(n2, n3) else f'{n1:.3e}'
		strf2 = '\\textbf{'+f'{n2:.3e}'+'}' if n2 < min(n1, n3) else f'{n2:.3e}'
		strf3 = '\\textbf{'+f'{n3:.3e}'+'}' if n3 < min(n1, n2) else f'{n3:.3e}'
		return strf1, strf2, strf3
	
	def __asterisk(self, n1: int, n2: int, s1: str, s2: str) -> tuple:
		self.ws1x3 += int(n1 < n2)
		self.ws3x1 += int(n2 < n1)
		s1 = f'${s1}'+'^{\\dagger}$' if n1 < n2 else s1
		s2 = f'${s2}'+'^{\\dagger}$' if n2 < n1 else s1
		return s1, s2

	def __underline(self, n1: int, n2: int, s1: str, s2: str) -> tuple:
		self.ws2x3 += int(n1 < n2)
		self.ws3x2 += int(n2 < n1)
		s1 = '\\underline{'+f'{s1}'+'}' if n1 < n2 else s1
		s2 = '\\underline{'+f'{s2}'+'}' if n2 < n1 else s2
		return s1, s2

	def __timeGap(self, t1: float, t2: float) -> float:
		mn, mx = min(t1, t2), max(t1, t2)
		gap = 100 * (1 - mn / mx)
		if t2 < t1: gap = -gap
		return gap

	def __header(self) -> str:
		return str().join(HEADER)

	def __row(self) -> str:
		mt1, mf1, bf1, sf1 = self.alg1.values()
		mt2, mf2, bf2, sf2 = self.alg2.values()
		mt3, mf3, bf3, sf3 = self.alg3.values()

		strbf1, strbf2, strbf3 = self.__textbf(bf1[0], bf2[0], bf3[0])
		strmf1, strmf2, strmf3 = self.__textbf(mf1[0], mf2[0], mf3[0])
		strsf1, strsf2, strsf3 = self.__textbf(sf1[0], sf2[0], sf3[0])

		if self.sig1: strmf1, strmf3 = self.__asterisk(mf1[0], mf3[0], strmf1, strmf3)
		#else: strmf1, strmf3 = strmf1+'\\hspace{1pt}', strmf3+'\\hspace{1pt}'
		if self.sig2: strmf2, strmf3 = self.__underline(mf2[0], mf3[0], strmf2, strmf3)

		row = f'\t\t{self.fun} & {strbf1} & {strmf1} & {strsf1} & {mt1[0]:.3e} &\n'
		row += f'\t\t& {strbf2} & {strmf2} & {strsf2} & {mt2[0]:.3e} &\n'
		row += f'\t\t& {strbf3} & {strmf3} & {strsf3} & {mt3[0]:.3e} &\n'
		row += f'\t\t& {self.__timeGap(mt1[0], mt3[0]):.2f}'
		row += f'& {self.__timeGap(mt2[0], mt3[0]):.2f}'
		return row+'\\\\\n'

	def __footer(self) -> str:
		footer = '\t\t\\hline\n\t\t\\multirow{2}{*}{\\textbf{\\# Wins (Sig.)}}'
		for key in ['best_fit', 'mean_fit', 'std_fit', 'mean_time']:
			gambiarra = f' ({self.ws1x3})' if key == 'mean_fit' else ''
			footer += ' & \\multirow{2}{*}{'+str(self.alg1[key][1])+gambiarra+'}'
		footer += '\n\t\t&'
		for key in ['best_fit', 'mean_fit', 'std_fit', 'mean_time']:
			gambiarra = f' ({self.ws2x3})' if key == 'mean_fit' else ''
			footer += ' & \\multirow{2}{*}{'+str(self.alg2[key][1])+gambiarra+'}'
		footer += '\n\t\t&'
		for key in ['best_fit', 'mean_fit', 'std_fit', 'mean_time']:
			gambiarra = f' ({self.ws3x1}/{self.ws3x2})' if key == 'mean_fit' else ''
			footer += ' & \\multirow{2}{*}{'+str(self.alg3[key][1])+gambiarra+'}'
		footer += '\n\t\t& & \\multirow{2}{*}{'+f'{self.mtg1/30:.2f}'+'}'
		footer += ' & \\multirow{2}{*}{'+f'{self.mtg2/30:.2f}'+'}'+f' \\\\{FOOTER}'
		return footer

# Main
if __name__ == '__main__':
	# Tabela
	os.makedirs('./output/tables/', exist_ok=True)
	table = Table(f'./output/tables/table_{args.alg1}x{args.alg2}x{args.alg3}.tex')

	# Loop das Funções
	for fun in range(1, 31):
		# Diretórios
		root = Path(f'./output/logs/{fun}')
		datas = {args.alg1: list(), args.alg2: list(), args.alg3: list()}
		files = [path for path in root.iterdir() if path.is_file() and path.suffix == '.log']

		# Arquivos
		for file in files:
			_, v, p = file.name.removesuffix('.log').split('-')
			if v == args.v and p in [args.alg1, args.alg2, args.alg3]:
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
		df3 = pd.DataFrame(datas[args.alg3], columns=['Time', 'Fit'])
		
		# Atualização
		table.update(fun, df1, df2, df3)
