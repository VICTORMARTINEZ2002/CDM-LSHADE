# Anotações

## Ideia
```c
int main(int argc, char **argv) {
    Inicializa_MPI;
    if (p > 1) {
        if (rank != 0) { // Processos Escravos
            Inicializa_LSHADE;
            while (...) { // Loop Principal
                ...
                if (...) { // Etapa de Mineração
                    Envia_Elite_para_Mestre;
                }
                if (...) { // Recebeu do Mestre (Iprobe)
                    Adiciona_Patterns_na_Populacao;
                }
                ...
            }
            Envia_Melhor_Indivíduo_para_Mestre;
        } else { // Processo Mestre
            while (...) { // Não Finaliza
                if (...) { // Recebeu Elite (Iprobe)
                    Adiciona_Elite_para_Minerar;
                }
                if (...) { // Pode Minerar
                    Minera_Todas_Elite;
                    Envia_Patterns_para_Todos_Escravos;
                }
            }
            Gera_Arquivo_Log;
        }
    } else {
        alg->run()
    }
    Finaliza_MPI;
}
```

## Main
- Entrada
    - Número da Função: 1-30
    - Número de Variáveis: 10/30/50/100
    - Número de Processos: 1/2/4/8/16
- Saída
    - Arquivo: {fun}-{var}-{p}.log
    - Efeito: Adiciona último tempo/best

## Report
- Entrada
    - Diretório de Logs
- Saída
    - Arquivo: report.txt
