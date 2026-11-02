#!/usr/bin/env bash
# One-off launcher for the stress campaign: each stage retried until done (all are resumable).
cd /home/nvidia/f_scarfato/runner || exit 1
retry() { for i in $(seq 1 600); do "$@" && return 0; echo "[campaign] stopped (board busy?), retry in 60 s"; sleep 60; done; return 1; }
echo "start $(date)"
retry ./measure/reporter.sh calibration && echo "reporter calibration done $(date)"
retry ./measure/reporter.sh networks    && echo "reporter networks done $(date)"
retry ./measure/sweep.sh --exp stress-sensitivity --static-chunks 8 --runs 5 && echo "sensitivity done $(date)"
retry ./measure/sweep.sh --exp stress-validation  --static-chunks 8 --runs 5 && echo "validation done $(date)"
echo "end $(date)"
