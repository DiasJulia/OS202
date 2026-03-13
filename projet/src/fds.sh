#!/bin/bash

# Nome do arquivo onde tudo será salvo
OUTPUT_FILE="resultados_benchmark.txt"

# Limpa o arquivo caso ele já exista
> $OUTPUT_FILE

echo "Iniciando a bateria de testes..." | tee -a $OUTPUT_FILE

comandos=(
    "OMP_NUM_THREADS=2 mpirun -np 1 ./ant_simu_v4.exe"
    "OMP_NUM_THREADS=4 mpirun -np 1 ./ant_simu_v4.exe"
    "OMP_NUM_THREADS=8 mpirun -np 1 ./ant_simu_v4.exe"
    "OMP_NUM_THREADS=1 mpirun -np 2 ./ant_simu_v4.exe"
    "OMP_NUM_THREADS=1 mpirun -np 4 ./ant_simu_v4.exe"
    "OMP_NUM_THREADS=1 mpirun -np 8 ./ant_simu_v4.exe"
    "OMP_NUM_THREADS=8 mpirun -np 2 ./ant_simu_v4.exe"
    "OMP_NUM_THREADS=8 mpirun -np 4 ./ant_simu_v4.exe"
    "OMP_NUM_THREADS=8 mpirun -np 8 ./ant_simu_v4.exe"
)

for cmd in "${comandos[@]}"; do
    echo "" | tee -a $OUTPUT_FILE
    echo "============================================================" | tee -a $OUTPUT_FILE
    echo "EXECUTANDO: $cmd" | tee -a $OUTPUT_FILE
    echo "============================================================" | tee -a $OUTPUT_FILE
    
    # Executa o comando, salva a saída no arquivo e mostra na tela ao mesmo tempo
    eval $cmd 2>&1 | tee -a $OUTPUT_FILE
done

echo "Todos os testes foram concluídos com sucesso! Verifique o arquivo $OUTPUT_FILE."