#!/usr/bin/env bash
# gpu-freq.sh <Hz> - lock the GPU clock at one of its available frequencies. Root: the user's step.
#
# The frequency experiment (the professor, 2026-11-10) locks lower GPU clocks with devfreq, in the MAXN
# power mode, rather than with a lower power mode: the 15 W and 30 W modes also power-gate TPCs (6 and 8
# SMs left, /etc/nvpmodel.conf), which would change the partitions under test. The memory and CPU clocks
# stay where jetson_clocks put them. `sudo jetson_clocks` restores the maximum.
#
# min_freq may never exceed max_freq, so going up raises max first and going down lowers min first.
set -eu
D=/sys/class/devfreq/17000000.gpu
F="${1:?usage: sudo $0 <Hz>, one of: $(cat $D/available_frequencies)}"
grep -qw -- "$F" $D/available_frequencies || { echo "not available: $F; one of: $(cat $D/available_frequencies)" >&2; exit 1; }
if [ "$F" -ge "$(cat $D/max_freq)" ]; then
    echo "$F" > $D/max_freq; echo "$F" > $D/min_freq
else
    echo "$F" > $D/min_freq; echo "$F" > $D/max_freq
fi
echo "GPU clock: min $(cat $D/min_freq) max $(cat $D/max_freq) now $(cat $D/cur_freq) (governor $(cat $D/governor))"
