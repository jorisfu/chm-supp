#!/usr/bin/bash
hyperfine \
  --runs 5 \
  --warmup 3 \
  -L dist euclidean,dixon_pds_sqeuclidean,nandist_euclidean \
  --export-json "./results/time/hyperfine-clusteronly.json" \
  './.venv/bin/python3 ./experiments/time/clusteronly.py {dist}'
