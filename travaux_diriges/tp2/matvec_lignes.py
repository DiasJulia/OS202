# Produit matrice-vecteur v = A.u
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

# Paralléliser le code séquentiel matvec.py en veillant à ce que chaque tâche 
# n’assemble que la partie de la matrice utile à sa somme partielle du produit matrice-vecteur. 
# On s’assurera que toutes les tâches à la fin du programme contiennent le vecteur résultat complet.

rank = comm.Get_rank()
size = comm.Get_size()
# Chaque tâche calcule sa portion locale de la matrice A
Time_start_parallel = MPI.Wtime()
A_local = A[rank*Nloc:(rank+1)*Nloc, :]
#print(f"A_local (tâche {rank}) = {A_local}")

# Chaque tâche calcule sa portion locale du produit matrice-vecteur
v_local = A_local.dot(u)
#print(f"v_local (tâche {rank}) = {v_local}")

# Réduction pour obtenir le vecteur résultat complet
result = np.zeros(dim)
comm.Gather(v_local, result)
Time_end_parallel = MPI.Wtime()

print(f"Temps de calcul parallèle: {Time_end_parallel - Time_start_parallel} secondes (tâche {rank})")
#print(f"v (tâche {rank}) = {result}")