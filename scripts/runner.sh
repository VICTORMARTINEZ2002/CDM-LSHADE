#!/bin/bash


runs=$(seq 1 20)
n_values=(1 2 4 6 8)
diversd_values=(1 0)
funcao_values=$(seq 1 30)
maxvar_values=(10 20 30 50 100)

make build
clear

# if [ -d "./output/logs" ]; then
# 	rm -rf ./output/logs/*
# fi
mkdir -p ./output/report
mkdir -p ./output/logs


for funcao in $funcao_values; do
	for n in "${n_values[@]}"; do
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



# for funcao in $funcao_values; do
#	python3 ./scripts/report.py 9
# done
