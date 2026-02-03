# Produit matrice-vecteur v = A.u (Parallélisation par COLONNES)
import numpy as np
import time
from mpi4py import MPI

# Dimension du problème (peut-être changé)
dim = 12000

# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
#print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
#print(f"u = {u}")

# Nombre de tâches
comm = MPI.COMM_WORLD
Nloc = dim // comm.Get_size()

# Paralléliser le code séquentiel matvec.py
rank = comm.Get_rank()
size = comm.Get_size()

Time_start_parallel = MPI.Wtime()

A_local = A[:, rank*Nloc:(rank+1)*Nloc]
#print(f"A_local shape (tâche {rank}) = {A_local.shape}") # Deve ser (120, Nloc)

u_local = u[rank*Nloc:(rank+1)*Nloc]

# Chaque tâche calcule sa portion locale du produit matrice-vecteur
v_partiel = A_local.dot(u_local)
#print(f"v_partiel (tâche {rank}) = {v_partiel}")

result = np.zeros(dim)
comm.Allreduce(v_partiel, result, op=MPI.SUM)

Time_end_parallel = MPI.Wtime()

print(f"Temps de calcul parallèle: {Time_end_parallel - Time_start_parallel} secondes (tâche {rank})")
#print(f"v (tâche {rank}) = {result}")