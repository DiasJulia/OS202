import numpy as np
from mpi4py import MPI

data = [round(x, 3) for x in np.random.standard_normal(1000).tolist()]
time_start = MPI.Wtime()
data = sorted(data)
time_end = MPI.Wtime()
print(f"Time taken for sorting: {time_end - time_start} seconds")