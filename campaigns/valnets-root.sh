#!/usr/bin/env bash
# valnets-root.sh - the counter model's profiles of the p99 networks; needs root.
# Run as root on board 1 with nothing else on the GPU:
#   cd /mnt/nvme/f.scarfato/scratch && bash valnets-root.sh 2>&1 | tee valnets-root.log
# 1. DRAM counters, all ten networks alone in an 8-SM MPS static partition (10 short arms, ~10 min);
# 2. ncu L2 traffic and hit rate, the six new networks alone on 4 SMs (green context: ncu replays one
#    kernel per MPS client, so this profile cannot run under MPS; ~10-20 min).
set -eu
R=/home/nvidia/f_scarfato/runner
cd "$R"
measure/profile.sh soc-mps
measure/ncu.sh nets-new
# the files are root's after this; give them back so the analysis can rewrite its derived CSVs
chown -R nvidia:nvidia /mnt/nvme/f.scarfato/profiles/soc-mps /mnt/nvme/f.scarfato/profiles/soc-new /mnt/nvme/f.scarfato/profiles/ncu/nets
echo VALNETS-ROOT-END
