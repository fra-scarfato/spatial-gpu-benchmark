#!/usr/bin/env bash
# Stops fs-b1b.sh as soon as its diagnostic is done (before the Burst-scrambled campaign starts), then
# runs the governor A/B under performance. The campaign is relaunched later, under the default governor.
S=/mnt/nvme/f.scarfato/scratch
until grep -q "DIAG-DONE\|FAILED" $S/fs-b1b.log; do sleep 1; done
P=$(pgrep -f "^bash fs-b1b.sh"); [ -n "$P" ] && kill $P
sleep 15   # let the smoke test's timeout (10 s) finish its itfproc
echo "[$(date '+%F %T')] stopped by the gate after DIAG-DONE: the campaign waits for the governor A/B" >> $S/fs-b1b.log
pgrep -fa "itfproc|reporter.sh" | grep -v pgrep >> $S/fs-gate.log
cd $S && GOV=performance bash fs-gov.sh > fs-gov-performance.log 2>&1
