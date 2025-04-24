#!/bin/bash

runs=$(seq 1 5)
n_values=(1 8 6 4 2)
diversd_values=(1 0)
funcao_values=$(seq 1 30)
maxvar_values=(100 50 20 10)

make build
clear

mkdir -p ./output/report
mkdir -p ./output/logs

for n in "${n_values[@]}"; do
	for funcao in $funcao_values; do
		for maxvar in "${maxvar_values[@]}"; do
			for divrsd in $diversd_values; do
				for run in $runs; do
					echo ">> Exec: Diversidade=$divrsd | FUNCAO=$funcao | n=$n | MAXVAR=$maxvar | run=$run"
					make run n=$n FUNCAO=$funcao MAXVAR=$maxvar DIVERSD=$divrsd SCRIPTF=1

				done
			done
		done
	done
done
