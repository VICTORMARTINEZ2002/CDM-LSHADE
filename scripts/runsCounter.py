import os
import re

def contar_linhas(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def gerar_relatorio(root_dir, output_file, arquivos):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.log'):
                arquivos.append(os.path.join(root, file))
    arquivos.sort(key=lambda x: [int(i) for i in re.findall(r'\d+', os.path.basename(x))])
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for arquivo in arquivos:
            num_linhas = contar_linhas(arquivo)
            out_file.write(f'{os.path.basename(arquivo):<12} -> {num_linhas}\n')

if __name__ == '__main__':
    arquivos = []
    root_dir = '../output/logs'
    output_file = './runsCounter.txt'
    gerar_relatorio(root_dir, output_file, arquivos)
