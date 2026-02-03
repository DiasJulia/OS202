# Produit matrice-vecteur v = A.u
import numpy as np
import time

# Dimension du problème (peut-être changé)
dim = 12000
# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
#print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
#print(f"u = {u}")

# Produit matrice-vecteur
Time_start = time.time()
v = A.dot(u)
Time_end = time.time()
print(f"Temps de calcul séquentiel: {Time_end - Time_start} secondes")
#print(f"v = {v}")
