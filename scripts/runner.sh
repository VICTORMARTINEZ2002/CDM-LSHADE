#!/bin/bash

runs=$(seq 1 15)
n_values=(1 2 3 4 5 6  8 9 10)
diversd_values=(1)
funcao_values=$(seq 1 30)
maxvar_values=(100 50 30 20 10)

mkdir -p ./output/report
mkdir -p ./output/logs
make build

for run in $runs; do
	for n in "${n_values[@]}"; do
		for funcao in $funcao_values; do
			for maxvar in "${maxvar_values[@]}"; do
				for divrsd in "${diversd_values[@]}"; do		
					echo ">> Exec: Diversidade=$divrsd | FUNCAO=$funcao | n=$n | MAXVAR=$maxvar | run=$run"
					make run n=$n FUNCAO=$funcao MAXVAR=$maxvar DIVERSD=$divrsd SCRIPTF=1
				done
			done
		done
	done
done
