#!/usr/bin/bash
hyperfine \
  --runs 50 \
  --warmup 3 \
  -L dist euclidean,dixon_pds_sqeuclidean,nandist_euclidean,eirola_esd_mvn,eirola_esd_gmm,mesquita_eed \
  --export-json "./results/time/hyperfine-clusteronly.json" \
  './.venv/bin/python3 ./experiments/time/clusteronly.py {dist}'
