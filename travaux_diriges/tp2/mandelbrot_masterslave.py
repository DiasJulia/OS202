# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

# --------------------------------------------------------------------------
# STRATÉGIE MAÎTRE-ESCLAVE (MASTER-SLAVE)
# --------------------------------------------------------------------------

if rank == 0:
    # --- Code du Maître (Master) ---
    print("Début du calcul Maître-Esclave...")
    deb = time()
    
    # Le maître détient l'image finale
    # Note: Dimensions (width, height) conformes au code original
    convergence = np.empty((width, height), dtype=np.double)
    
    currentRow = 0
    active_workers = size - 1 # Tous les processus sauf le 0
    
    # 1. Distribution initiale : envoyer une ligne à chaque esclave
    for p in range(1, size):
        if currentRow < height:
            comm.send(currentRow, dest=p, tag=1) # Tag 1 = Travail
            currentRow += 1
        else:
            comm.send(-1, dest=p, tag=1) # Tag 1 avec -1 = Arrêt
            active_workers -= 1

    # 2. Boucle de gestion dynamique
    status = MPI.Status()
    while active_workers > 0:
        # Recevoir le résultat de n'importe quel esclave
        # On reçoit un tuple : (index_ligne, données_calculées)
        row_idx, row_data = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        source = status.Get_source()
        
        # Stocker le résultat dans la matrice globale
        convergence[:, row_idx] = row_data
        
        # Envoyer la prochaine tâche à cet esclave spécifique
        if currentRow < height:
            comm.send(currentRow, dest=source, tag=1)
            currentRow += 1
        else:
            # Plus de travail, on libère l'esclave
            comm.send(-1, dest=source, tag=1)
            active_workers -= 1
            
    fin = time()
    print(f"Temps total (Calcul + Constitution) : {fin-deb}")

    # Constitution de l'image résultante :
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    image.show()

else:
    # --- Code des Esclaves (Slaves) ---
    # Boucle infinie : Demander travail -> Calculer -> Renvoyer -> Répéter
    while True:
        # Recevoir l'index de la ligne à calculer (Tag 1)
        y = comm.recv(source=0, tag=1)
        
        # Si on reçoit -1, c'est le signal d'arrêt (Poison Pill)
        if y == -1:
            break
            
        # Sinon, on calcule la ligne
        row_data = np.empty(width, dtype=np.double)
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            row_data[x] = mandelbrot_set.convergence(c, smooth=True)
            
        # Renvoyer le résultat au maître (Tag 2)
        # On renvoie (y, row_data) pour que le maître sache où placer les données
        comm.send((y, row_data), dest=0, tag=2)