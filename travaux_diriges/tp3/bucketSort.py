import random
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

def bucket_division(arr, bucket_count):
    if len(arr) == 0:
        return []
    
    min_value = min(arr)
    max_value = max(arr)

    buckets = [[] for _ in range(bucket_count)]

    for num in arr:
        bucket_index = int((num - min_value) / (max_value - min_value + 1e-5) * bucket_count)
        if bucket_index == bucket_count:
            bucket_index -= 1
        buckets[bucket_index].append(num)

    #print(f"Process {comm.Get_rank()} created buckets: {buckets}")
    
    return buckets

def bucket_sort(arr, bucket_count):
    time_start_total = MPI.Wtime()
    buckets = None
    if comm.Get_rank() == 0:
        buckets = bucket_division(arr, bucket_count)
    bucket = comm.scatter(buckets, root=0)
    #print(f"Process {comm.Get_rank()} received bucket: {bucket}")
    print(f"Process {comm.Get_rank()} received bucket with {len(bucket)} elements.")
    time_start = MPI.Wtime()
    bucket = sorted(bucket)

    #print(f"Process {comm.Get_rank()} sorted its bucket: {local_sorted_bucket
    all_sorted_buckets = comm.gather(bucket, root=0)
    time_end = MPI.Wtime()
    print(f"Time {comm.Get_rank()} took for sorting: {time_end - time_start} seconds")

    if comm.Get_rank() == 0:
        pass
        #print(f"Process {comm.Get_rank()} gathered sorted buckets: {all_sorted_buckets}")
    time_end_total = MPI.Wtime()
    print(f"Process {comm.Get_rank()} total time taken: {time_end_total - time_start_total} seconds")

# Example usage:
if __name__ == "__main__":
    data = [round(x, 3) for x in np.random.standard_normal(10000).tolist()]
    bucket_sort(data, bucket_count=comm.Get_size())