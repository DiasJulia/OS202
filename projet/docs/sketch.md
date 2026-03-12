# Support document to help in the writing of the report. 
It contains a sketch of the main points to be covered, and some notes on the implementation.

# First benchmarking experiments

## Integrating benchmarking code into the simulation
- Implemented a benchmark mode in `ant_simu.cpp` that runs the simulation for a specified number of iterations without rendering, and measures the average time taken for the advance and display steps separately.
- The benchmark mode can be activated with the `--benchmark` flag
- The code can be executed in benchmark mode with either the vectorized or OOP implementation by using the `--benchmark--oop` flag for the OOP version.
- The results are printed at the end of the simulation, showing the total time taken, average time per iteration, and the average times for the advance and display steps.

## Profiling with gprof
- The `Makefile` has been updated to include a `profile` target that runs the benchmark and generates gprof reports for both implementations.
- The profiling results can be found in `profile_vectorized.txt` and `profile_oop.txt` after running the `make profile` command.

## Initial benchmarking results
- The initial benchmarks show that the vectorized implementation is significantly faster than the OOP version
- The following results were obtained for 5000 iterations:

Profiling vectorized mode...
Mode vectorisé (defaut)
La première nourriture est arrivée au nid a l'iteration 1996
[benchmark] mode=vectorized iters=5000
 total_ms=93352
 ms_per_iter=18.6704
 avg_advance_ms=2.13287
 avg_display_ms=16.5207
 food=860
Profiling oop mode...
Mode POO (--oop)
La première nourriture est arrivée au nid a l'iteration 1957
[benchmark] mode=oop iters=5000
 total_ms=92489.4
 ms_per_iter=18.4979
 avg_advance_ms=2.17602
 avg_display_ms=16.3055
 food=488
Reports: profile_vectorized.txt, profile_oop.txt

- More detailed analysis of the profiling results will be needed to identify the specific bottlenecks in each implementation. Details are available in the generated gprof reports.