import numpy as np
import bisect
from mpi4py import MPI

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

total_len = 10000
local_len = total_len // size

if __name__ == "__main__":
    buckets = [[] for _ in range(size)]

    arr = np.random.standard_normal(local_len).tolist()

    #print(f"Process {rank} initial array:", arr)

    time_start = MPI.Wtime()
    # Get quantiles to define bucket ranges
    quantiles = np.percentile(arr, np.linspace(0, 100, size + 1))

    # Gather quantiles for all processes
    all_quantiles = comm.allgather(quantiles)
    all_quantiles = np.sort(np.concatenate(all_quantiles))

    #print(f"Process {rank} all quantiles:", all_quantiles)

    idx = np.linspace(0, len(all_quantiles)-1, size+1).astype(int)
    all_quantiles = all_quantiles[idx]

    # Distribute elements into buckets based on quantiles
    for num in arr:
        bucket_index = bisect.bisect_right(all_quantiles, num) - 1
        if bucket_index == size:
            bucket_index -= 1
        buckets[bucket_index].append(num)
    
    buckets = comm.alltoall(buckets)

    buckets = [bucket for sublist in buckets for bucket in sublist]

    #print(f"Process {rank} received buckets:", buckets)

    # Each process sorts its own bucket
    buckets = sorted(buckets)

    # Gather all sorted buckets at root
    all_sorted_buckets = comm.gather(buckets, root=0)
    
    #if rank == 0:
        # Merge all sorted buckets
        #print("Final sorted array:", [round(item, 2) for sublist in all_sorted_buckets for item in sublist])

    time_end = MPI.Wtime()
    print(f"Process {rank} time taken for sorting: {time_end - time_start} seconds")

