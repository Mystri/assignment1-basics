# Lecture 1

# Lecture 2 Modules in LLM
GPU - Lecture 5, 16min
1. VS CPU: less control, more computation.
Why parallelism: optimized for **throughput**, compared to **latency**.
2. Hardware perspective
SP < SM < Chip
 - SM = Unit of control, holding a lot of SPs. Containing:
   - Fastest memory: L1 Cache, Shared memory.
   - SP can do matrix multiplication. 

Chip: L2 Cache.

3. Threads < Blocks < Warps
 - Threads. 
   - Threads has: local -> shared(across threads) -> global(across blocks[DRAM   ]) memory.
   - Threads executes the same code with different input (SIMT).

Other structures:
TPU
tensore core ~= SM, containing:
 - scalar unit ~= Cpu, dispatching instructions to the rest.
 - VPU - performs elementwise operations; load data to mxu.
 - MXU - matrix mult
  
Problem: optimize GPU so that it constantly runs some operations, instead of waiting for memory throughput.
Roofline model. 


# Lecture 5 GPU Architecture