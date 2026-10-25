"""One resource manager for every GPU we may use on this machine, and every job that needs one.

Before: each study, queue and benchmark started its own scheduler, they only excluded each other
through GPU leases, no one saw the whole queue, and jobs asked for fixed GPU ids -- so GPUs sat idle
while work waited elsewhere (2026-10-25: GPUs 1 and 3 idle for half an hour while a benchmark waited
for a study process to exit).

Now: one daemon (scripts/pool.py serve) owns paths.SCHEDULED_GPUS. Jobs are submitted, not run:
each declares a shape (a GPU count, or an elastic min..max), a priority, whether it can be preempted,
dependencies and a done check -- never GPU ids. The daemon matches ready jobs to free GPUs every few
seconds (allocate.decide: gang allocation, priority, preemption of lower-priority preemptible jobs,
backfill of free GPUs with preemptible work), launches each job with CUDA_VISIBLE_DEVICES set to its
GPUs (so every job sees GPUs 0..n-1 and "{gpus}" in its commands becomes "0 1 ..."), takes the GPU
leases other schedulers respect, and records everything under results/pool/.

  allocate  the matching decision (pure, tested)
  store     job records (results/pool/jobs/), the inbox new jobs are dropped into, done checks
  daemon    the loop: ingest, reap, decide, preempt, launch
  run       the per-job process: run its commands in order, appending to its log
"""
