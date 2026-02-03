# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm

from mpi4py import MPI

comm = MPI.COMM_WORLD

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

nLines = height // comm.Get_size()
rank = comm.Get_rank()

# Chaque tâche calcule sa portion locale de l'image
deb_parallel = MPI.Wtime()
convergence_local = np.empty((nLines, width), dtype=np.double)

scaleX = 3./width
scaleY = 2.25/height
# Calcul de l'ensemble de mandelbrot :
for y in range(rank*nLines, (rank+1)*nLines):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        convergence_local[y - rank*nLines, x] = mandelbrot_set.convergence(c, smooth=True)

fin_parallel = MPI.Wtime()
print(f"Temps de calcul parallèle (tâche {rank}) : {fin_parallel - deb_parallel} secondes")


# Constitution de l'image résultante :
convergence = None
if rank == 0:
    convergence = np.empty((height, width), dtype=np.double)

comm.Gather(sendbuf=convergence_local, recvbuf=convergence, root=0)

if rank == 0:
    print("Constitution de l'image...")
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()
