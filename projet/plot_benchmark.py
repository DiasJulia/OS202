#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Dados do benchmark (fixos)
# =========================
data = [
    {
        "mode": "Original",
        "threads": 1,
        "avg_advance_ms": 2.74543,
        "avg_evap_ms": 0.483387,
        "avg_update_ms": 0.00755527,
        "avg_display_ms": 10.1293,
        "avg_iter_ms": 16.8762,
        "iters": 3556,
    },
    {
        "mode": "Vectorized",
        "threads": 1,
        "avg_advance_ms": 2.34091,
        "avg_evap_ms": 0.393619,
        "avg_update_ms": 0.00604881,
        "avg_display_ms": 10.8273,
        "avg_iter_ms": 16.939,
        "iters": 3543,
    },
    {
        "mode": "OMP-2",
        "threads": 2,
        "avg_advance_ms": 2.06016,
        "avg_evap_ms": 0.503808,
        "avg_update_ms": 0.00779269,
        "avg_display_ms": 11.2452,
        "avg_iter_ms": 16.8782,
        "iters": 3555,
    },
    {
        "mode": "OMP-4",
        "threads": 4,
        "avg_advance_ms": 1.7087,
        "avg_evap_ms": 0.532912,
        "avg_update_ms": 0.00804289,
        "avg_display_ms": 11.684,
        "avg_iter_ms": 16.9192,
        "iters": 3547,
    },
    {
        "mode": "OMP-8",
        "threads": 8,
        "avg_advance_ms": 1.50871,
        "avg_evap_ms": 0.655052,
        "avg_update_ms": 0.00815751,
        "avg_display_ms": 12.0674,
        "avg_iter_ms": 17.0359,
        "iters": 3523,
    },
]

# =========================
# Pre-processamento
# =========================
modes = [d["mode"] for d in data]
advance = np.array([d["avg_advance_ms"] for d in data], dtype=float)
evap = np.array([d["avg_evap_ms"] for d in data], dtype=float)
update = np.array([d["avg_update_ms"] for d in data], dtype=float)
display = np.array([d["avg_display_ms"] for d in data], dtype=float)
iteration = np.array([d["avg_iter_ms"] for d in data], dtype=float)
iters = np.array([d["iters"] for d in data], dtype=float)

# Throughput aproximado (iter/s)
throughput = 1000.0 / iteration

# Breakdown estimado
# "compute_core" = advance - evap - update
# "other_iter_overhead" = iteration - (advance + display)
compute_core = advance - evap - update
other_overhead = iteration - (advance + display)

# Subconjunto OMP para speedup/eficiencia
omp_data = [d for d in data if d["mode"].startswith("OMP")]
omp_threads = np.array([d["threads"] for d in omp_data], dtype=float)
omp_adv = np.array([d["avg_advance_ms"] for d in omp_data], dtype=float)
omp_it = np.array([d["avg_iter_ms"] for d in omp_data], dtype=float)

# Baselines
# 1) baseline OMP-2 (comparacao intra-OMP)
omp2_adv = omp_adv[0]
omp2_it = omp_it[0]
speedup_adv_vs_omp2 = omp2_adv / omp_adv
speedup_it_vs_omp2 = omp2_it / omp_it

# 2) baseline Vectorized (comparacao com v2)
vec = next(d for d in data if d["mode"] == "Vectorized")
speedup_adv_vs_vec = vec["avg_advance_ms"] / omp_adv
speedup_it_vs_vec = vec["avg_iter_ms"] / omp_it

eff_adv_vs_omp2 = speedup_adv_vs_omp2 / (omp_threads / omp_threads[0])
eff_it_vs_omp2 = speedup_it_vs_omp2 / (omp_threads / omp_threads[0])

# =========================
# Plot
# =========================
plt.style.use("seaborn-v0_8-whitegrid")
out_dir = "plots_benchmark_aco"
os.makedirs(out_dir, exist_ok=True)

# 1) Barras: tempos principais por modo
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(modes))
w = 0.38
ax.bar(x - w/2, advance, width=w, label="avg advance_time (ms)")
ax.bar(x + w/2, iteration, width=w, label="avg iteration (ms)")
ax.set_xticks(x)
ax.set_xticklabels(modes, rotation=20)
ax.set_ylabel("Tempo medio (ms)")
ax.set_title("Tempo medio por iteracao e por advance_time")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "01_tempos_principais.png"), dpi=160)

# 2) Breakdown empilhado da iteracao
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x, compute_core, label="ant compute (advance - evap - update)")
ax.bar(x, evap, bottom=compute_core, label="evaporation")
ax.bar(x, update, bottom=compute_core + evap, label="pheromone update")
ax.bar(x, display, bottom=compute_core + evap + update, label="display")
ax.bar(x, other_overhead, bottom=compute_core + evap + update + display, label="other overhead")
ax.set_xticks(x)
ax.set_xticklabels(modes, rotation=20)
ax.set_ylabel("Tempo medio (ms)")
ax.set_title("Breakdown da iteracao media")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "02_breakdown_iteracao.png"), dpi=160)

# 3) Speedup OMP
fig, ax = plt.subplots(figsize=(8, 5))
ideal = omp_threads / omp_threads[0]  # ideal vs OMP-2
ax.plot(omp_threads, speedup_adv_vs_omp2, "o-", label="Speedup advance vs OMP-2")
ax.plot(omp_threads, speedup_it_vs_omp2, "s-", label="Speedup iteration vs OMP-2")
ax.plot(omp_threads, ideal, "--", label="Ideal linear (baseline OMP-2)")
ax.set_xlabel("Threads OMP")
ax.set_ylabel("Speedup")
ax.set_title("Speedup OMP (baseline = OMP-2)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "03_speedup_omp.png"), dpi=160)

# 4) Eficiencia OMP
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(omp_threads, eff_adv_vs_omp2, "o-", label="Eficiencia advance")
ax.plot(omp_threads, eff_it_vs_omp2, "s-", label="Eficiencia iteration")
ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
ax.set_xlabel("Threads OMP")
ax.set_ylabel("Eficiencia paralela")
ax.set_title("Eficiencia OMP (baseline = OMP-2)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "04_eficiencia_omp.png"), dpi=160)

# 5) Throughput
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(modes, throughput)
ax.set_ylabel("Iteracoes por segundo (aprox.)")
ax.set_title("Throughput global por modo")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "05_throughput.png"), dpi=160)

# 6) Speedup OMP vs vectorized (v2)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(omp_threads, speedup_adv_vs_vec, "o-", label="Speedup advance vs Vectorized")
ax.plot(omp_threads, speedup_it_vs_vec, "s-", label="Speedup iteration vs Vectorized")
ax.set_xlabel("Threads OMP")
ax.set_ylabel("Speedup")
ax.set_title("Ganho da versao OMP sobre a versao vectorized (v2)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "06_speedup_vs_vectorized.png"), dpi=160)

print(f"Graficos salvos em: {out_dir}")
print("Arquivos:")
for f in sorted(os.listdir(out_dir)):
    print(" -", f)