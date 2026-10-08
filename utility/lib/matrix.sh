#!/bin/bash
# The run matrix: one `add` per arm, and the loops that generate the families.
# Sourced by sweep.sh after the command line, because some arms are conditional on it.
#
# A job is a name plus a spec of the form
#      kind | mps | envA | envB | cmdA | cmdB | ratio
#   kind : solo (one process) | pair (two processes) | itf (one measured, one co-runner)
#   mps  : off | on | static   (static also creates the partitions)
#   env* : extra environment for that client
#   cmd* : the invocation, without the protocol flags, which the driver adds
#   ratio: the target's percentage; the static arms use it to size the partitions, and a
#          colon ratio places three of them
#
# Pipe-separated and read back with `read`, which keeps empty fields unambiguous.

declare -a JOB_NAME=() JOB_SPEC=()
add() { JOB_NAME+=("$1"); JOB_SPEC+=("$2"); }

# Experiment 1: one network, isolated.
# `whole` is the baseline; `gc100` is NOT redundant — it is the control that
# proves the green-context mechanism itself costs nothing, so a slowdown at 75%
# cannot be blamed on the mechanism.
for net in mobilenet resnet; do
    if [ "$net" = mobilenet ]; then eng="$MOB"; else eng="$RES"; fi
    add "isolated-scaling_${net}_whole" "solo|off|||$RUNNER $eng $DS --allocation whole|"
    for p in 100 75 50 25; do
        add "isolated-scaling_${net}_gc${p}" \
            "solo|off|||$RUNNER $eng $DS --allocation greenctx --sm-percent $p|"
    done
done

# Experiment 2: two networks in ONE process, disjoint green contexts.
#
# `--submit-threads 2` is stated rather than left to the default. The runner's
# default is now ONE submitting thread, because that is the configuration that
# removes the host-side tail; the pair jobs here are the two-thread case that
# those results were measured with, and a re-run must reproduce them, not
# quietly measure something else.
#
# MobileNet + ResNet rather than mob-mob/res-res: that is the deployment being
# studied (a scheduler co-locates *different* workloads), and the mixed pair
# yields each network's co-run latency at 4, 8 and 12 SMs — exactly the points
# Experiment 1 measures in isolation.
add "isolated-scaling_inception_whole"      "solo|off|||$RUNNER $INC  $DS --allocation whole|"
add "isolated-scaling_inception_aux1_whole" "solo|off|||$RUNNER $INC1 $DS --allocation whole|"
add "isolated-scaling_inception_aux5_whole" "solo|off|||$RUNNER $INC5 $DS --allocation whole|"
add "isolated-scaling_vit_whole"            "solo|off|||$RUNNER $VIT  $DS --allocation whole|"
add "isolated-scaling_retinanet_whole"      "solo|off|||$RUNNER $RN  $DS --allocation whole|"
add "isolated-scaling_retinanet_aux1_whole" "solo|off|||$RUNNER $RN1 $DS --allocation whole|"
add "isolated-scaling_retinanet_aux8_whole" "solo|off|||$RUNNER $RN8 $DS --allocation whole|"
add "isolated-scaling_inception_aux2_whole"  "solo|off|||$RUNNER $INC2 $DS --allocation whole|"
add "isolated-scaling_inception_aux4_whole"  "solo|off|||$RUNNER $INC4 $DS --allocation whole|"
add "isolated-scaling_retinanet_aux16_whole" "solo|off|||$RUNNER $RN16 $DS --allocation whole|"

add "pair-greenctx-2submit_baseline" \
    "solo|off|||$RUNNER --submit-threads 2 --dataset $DS --allocation whole --net $MOB:100:mobilenet --net $RES:100:resnet|"

for split in 50:50 75:25 25:75; do
    mp="${split%%:*}"; rp="${split##*:}"
    add "pair-greenctx-2submit_${mp}_${rp}" \
        "solo|off|||$RUNNER --submit-threads 2 --dataset $DS --allocation greenctx --net $MOB:$mp:mobilenet --net $RES:$rp:resnet|"
done

# Experiment 3: MPS, TWO processes.
#
# Phase 1, the minimal comparison plus the control that makes it readable:
#   mps_single : one process under MPS, vs the isolated `whole` run
#   mps_off    : two processes, no MPS                  -> the 2-process baseline
#   mps_default: two processes under MPS                -> the co-scheduling effect
#
# On alignment: the two processes have no shared memory to rendezvous in, so
# each starts its timed phase when its own warm-up ends. The skew is the
# difference in pre-timed work — about 0.16 s (200 inferences: 0.11 s MobileNet
# vs 0.27 s ResNet) out of a 30 s window, so roughly 0.5%. A `--start-at`
# deadline would make it exact; it is not needed to see an effect of the size
# MPS would have to show in order to matter.
add "pair-mpsstatic-2submit_mps_single"  "solo|on|||$RUNNER $MOB $DS --allocation whole|"
add "pair-mpsstatic-2submit_mps_off"     "pair|off|||$RUNNER $MOB $DS --allocation whole|$RUNNER $RES $DS --allocation whole"
add "pair-mpsstatic-2submit_mps_default" "pair|on|||$RUNNER $MOB $DS --allocation whole|$RUNNER $RES $DS --allocation whole"

# Phase 2: active thread percentage. This is NOT spatial partitioning. ATP
# limits the fraction of the SM's thread slots a client may occupy, which scales
# throughput but does not reserve particular SMs.
for split in 50:50 75:25 25:75; do
    mp="${split%%:*}"; rp="${split##*:}"
    add "pair-mpsstatic-2submit_mps_atp_${mp}_${rp}" \
        "pair|on|CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mp|CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$rp|$RUNNER $MOB $DS --allocation whole|$RUNNER $RES $DS --allocation whole"
done

# Phase 3: static SM partitions, the actual spatial analogue of green contexts
# and the only part with an unknown. Documented behaviour: partitions are
# exclusive to their clients and cannot be over-allocated, and a client that
# does not set CUDA_MPS_SM_PARTITION FAILS to create a context — so a
# misconfigured run dies loudly instead of silently measuring nothing.
#
# Only added when --static-chunks is given, because the chunk count has to be
# read off `lspart` on the real board first.
if [ "$STATIC_CHUNKS" -gt 0 ]; then
    for split in 50:50 75:25 25:75; do
        mp="${split%%:*}"; rp="${split##*:}"
        add "pair-mpsstatic-2submit_mps_static_${mp}_${rp}" \
            "pair|static|||$RUNNER $MOB $DS --allocation whole|$RUNNER $RES $DS --allocation whole|$mp"
    done
fi

# The work queue experiment: Experiment 2's setup with the work queue half of
# the green context descriptor set EXPLICITLY instead of inherited.
#
# Experiment 2 never put a work queue resource in the descriptor at all, so every
# run of it used the driver default: all contexts sharing one set of queues, with
# a concurrency limit of CUDA_DEVICE_MAX_CONNECTIONS. Measured on this board that
# default is 8 normally and 2 under MPS — they are not the same, which is why
# this is an experiment rather than an assumption.
#
# The hypothesis is the guide's own warning: independent work on different
# streams that maps to the SAME work queue acquires a false dependence and is
# serialised even when SMs are free. MobileNet's kernels are short and numerous,
# so it submits far more often per unit of work than ResNet, and it is the
# network that should show the effect.
#
# Three knobs have to be distinguished, and only two of them are runner options:
#   --wq-scope  how the driver shares queues between contexts
#   --wq-limit  the concurrency declared to the driver as a HINT (not a stream
#               limit, and not a cap on anything)
#   CUDA_DEVICE_MAX_CONNECTIONS is a different thing again: it bounds how many
#   queues exist at all, is read from the environment before CUDA initialises,
#   and therefore lives in the env field of the spec. Both knobs are pinned in
#   every arm so the arms stay comparable.
#
# Five arms x two splits = the 2x2 factorial on (scope, limit) plus the
# limit-1 negative control. `devctx_8` must reproduce Experiment 2's numbers; if
# it does not, the code path is wrong and the rest is uninterpretable.
#
# Select with:  ./utility/sweep.sh --exp wq
for split in 50:50 75:25; do
    mp="${split%%:*}"; rp="${split##*:}"
    for arm in devctx:8:device_ctx balanced:8:green_ctx_balanced \
               devctx:2:device_ctx balanced:2:green_ctx_balanced \
               devctx:1:device_ctx; do
        tag="${arm%%:*}"; rest="${arm#*:}"
        lim="${rest%%:*}"; scope="${rest##*:}"
        add "pair-greenctx-workqueue-2submit_${tag}_${lim}_${mp}_${rp}" \
            "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --submit-threads 2 --dataset $DS --allocation greenctx --wq-scope $scope --wq-limit $lim --net $MOB:$mp:mobilenet --net $RES:$rp:resnet|"
    done
done

# The unpartitioned pair: no green context, so no scope applies and only the
# queue BUDGET is set. These are Experiment 2's `baseline` run twice — once at
# the limit Experiment 2 actually had (8) and once at the one Experiment 3 had
# (2). That pair measures the process model with the budget held equal, which the
# Exp 2 versus Exp 3 comparison cannot do because those two differ in both at
# once. It is the cleanest pair in the whole matrix and it costs 10 runs.
for lim in 8 2; do
    add "pair-greenctx-workqueue-2submit_unpart_${lim}" \
        "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --submit-threads 2 --dataset $DS --allocation whole --net $MOB:100:mobilenet --net $RES:100:resnet|"
done

# The auxiliary-stream family: Experiment 2's pair again, but with ResNet built
# to allow one auxiliary stream, so it issues two streams of its own.
#
# ResNet is the only engine here with any intra-network parallelism: TensorRT
# says so in its build log ("Number of aux streams is 1", 7 events per
# inference) and measured against a trace the two streams overlap only about
# 4.5% of the time. MobileNet reports "cannot find an opportunity to leverage
# it" under every cap, so it never has more than one stream.
#
# That gives the pair three concurrent stream-ordered workloads - ResNet's two
# plus MobileNet's one - so the declared concurrency can finally be chosen from
# the measured parallelism rather than guessed. The family is built around three
# questions:
#   - the sharing scope at the coherent concurrency (limit 3), which is the
#     controlled comparison the whole work queue argument rests on;
#   - the same comparison at the per-client reading of coherence (limit 2), to
#     settle whether the hint is read per context or device-wide;
#   - the sweep he asked for: 1, the observed concurrency, and the device's
#     default (8) and maximum (32).
# Limit 8 at device_ctx is the attribution arm: it meets Experiment 4's
# devctx_8 rows at the same limit, scope and ratio, so the difference there is
# the auxiliary stream and nothing else.
for split in 50:50 75:25; do
    mp="${split%%:*}"; rp="${split##*:}"
    for arm in devctx:3:device_ctx balanced:3:green_ctx_balanced \
               devctx:2:device_ctx balanced:2:green_ctx_balanced \
               balanced:1:green_ctx_balanced balanced:8:green_ctx_balanced \
               balanced:32:green_ctx_balanced devctx:8:device_ctx; do
        tag="${arm%%:*}"; rest="${arm#*:}"
        lim="${rest%%:*}"; scope="${rest##*:}"
        add "pair-greenctx-auxstreams-2submit_${tag}_${lim}_${mp}_${rp}" \
            "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --submit-threads 2 --dataset $DS --allocation greenctx --wq-scope $scope --wq-limit $lim --net $MOB:$mp:mobilenet --net $RESAUX:$rp:resnet|"
    done
done

# The aux-enabled engine on its own, no green context and nothing else changed:
# the reference its own pair rows are read against, and the row that says whether
# permitting the auxiliary stream changed ResNet's own latency at all.
add "pair-greenctx-auxstreams-2submit_aux1_whole" \
    "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=8||$RUNNER $RESAUX $DS --allocation whole|"

# ---------------------------------------------------------------------------
# The same three intra-process families, measured with ONE submitting
# thread.
#
# The pair has so far always been submitted by two threads, one per network.
# That is the naive application and it stays: every table above is built on it.
# It is also the source of the MobileNet tail, because two threads contend in
# the launch path of one CUDA context, a launch is occasionally delayed, and the
# GPU then idles inside the event window. MPS never showed the tail for the
# mirror-image reason: it is two processes, and two processes do not share that
# path.
#
# These arms change exactly one thing. A single thread drives both networks'
# timed loops, keeping one inference in flight per network — the same depth the
# threaded path has — so the two sets differ in the submitter and in nothing
# else: same engines, same green contexts, same joint SM split, same queue
# settings, same protocol. Measured effect on the pair at 75/25: MobileNet's
# tail falls from 0.348 to 0.050 ms and its median improves slightly, while
# throughput does not move.
#
# Select with:  ./utility/sweep.sh --exp 1t
ST="--submit-threads 1"

for split in 50:50 75:25 25:75; do
    mp="${split%%:*}"; rp="${split##*:}"
    add "pair-greenctx-1submit_${mp}_${rp}" \
        "solo|off|||$RUNNER --dataset $DS --allocation greenctx $ST --net $MOB:$mp:mobilenet --net $RES:$rp:resnet|"
done
add "pair-greenctx-1submit_baseline" \
    "solo|off|||$RUNNER --dataset $DS --allocation whole $ST --net $MOB:100:mobilenet --net $RES:100:resnet|"

for split in 50:50 75:25; do
    mp="${split%%:*}"; rp="${split##*:}"
    for arm in devctx:8:device_ctx balanced:8:green_ctx_balanced \
               devctx:2:device_ctx balanced:2:green_ctx_balanced \
               devctx:1:device_ctx; do
        tag="${arm%%:*}"; rest="${arm#*:}"
        lim="${rest%%:*}"; scope="${rest##*:}"
        add "pair-greenctx-workqueue-1submit_${tag}_${lim}_${mp}_${rp}" \
            "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation greenctx $ST --wq-scope $scope --wq-limit $lim --net $MOB:$mp:mobilenet --net $RES:$rp:resnet|"
    done
done
for lim in 8 2; do
    add "pair-greenctx-workqueue-1submit_unpart_${lim}" \
        "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation whole $ST --net $MOB:100:mobilenet --net $RES:100:resnet|"
done

for split in 50:50 75:25; do
    mp="${split%%:*}"; rp="${split##*:}"
    for arm in devctx:3:device_ctx balanced:3:green_ctx_balanced \
               devctx:2:device_ctx balanced:2:green_ctx_balanced \
               balanced:1:green_ctx_balanced balanced:8:green_ctx_balanced \
               balanced:32:green_ctx_balanced devctx:8:device_ctx; do
        tag="${arm%%:*}"; rest="${arm#*:}"
        lim="${rest%%:*}"; scope="${rest##*:}"
        add "pair-greenctx-auxstreams-1submit_${tag}_${lim}_${mp}_${rp}" \
            "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation greenctx $ST --wq-scope $scope --wq-limit $lim --net $MOB:$mp:mobilenet --net $RESAUX:$rp:resnet|"
    done
done
# One network, so this arm has a single submitting thread in both sets; it is
# repeated here only so the one-thread family stands on its own.
add "pair-greenctx-auxstreams-1submit_aux1_whole" \
    "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=8||$RUNNER $RESAUX $DS --allocation whole|"

# The submission model. One process, two networks, green contexts, 75/25 -- the
# split where the two-thread tail is largest. Four arms, one question: where do the
# submitting threads run, and at what priority.
#
#   t1_unpinned    a single submitter, no affinity            (the restructuring)
#   t1_pinfifo99   a single submitter on core 6, SCHED_FIFO 99
#   t2_unpinned    one thread per network, no affinity        (the naive application)
#   t2_core67      both submitters allowed cores 6 and 7
#
# `--cpu` and `--priority` are applied from INSIDE the process, so they confine the
# submitting thread alone; `taskset` confines every thread the process has, the CUDA
# driver's included. The mixture is deliberate: the two-thread arm is what an
# application can do with a stock tool, and the pinned one-thread arm is what it
# cannot, which is why that one is a flag here.
add "submission-model_t2_unpinned" \
    "solo|off|||$RUNNER --submit-threads 2 --dataset $DS --allocation greenctx --net $MOB:75:mobilenet --net $RES:25:resnet|"
add "submission-model_t1_unpinned" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:75:mobilenet --net $RES:25:resnet|"
add "submission-model_t2_core67" \
    "solo|off|||taskset -c $SUBMIT_CORE,$SUBMIT_CORE_B $RUNNER --submit-threads 2 --dataset $DS --allocation greenctx --net $MOB:75:mobilenet --net $RES:25:resnet|"

# The pinned-and-prioritised arm. 99 is the maximum RT priority the kernel offers;
# the submitting thread blocks on events rather than scpu-affinity, so it never approaches
# the RT runtime throttle, and the run records the policy it was actually granted.
if [ "$WITH_PRIORITY" = 1 ]; then
    add "submission-model_t1_pinfifo99" \
        "solo|off|||$RUNNER --submit-threads 1 --cpu $SUBMIT_CORE --priority fifo:99 --dataset $DS --allocation greenctx --net $MOB:75:mobilenet --net $RES:25:resnet|"
fi

# The one MPS arm. MPS runs one process per network, so each has exactly one
# submitter and the 1-thread/2-thread axis does not exist there. What can vary across
# processes is where each submitter runs: giving each its own core is the
# cross-process analogue of the green-context cpu-affinity, and that is the whole of it.
#
# Needs --static-chunks, like every static arm: the partition count is read off
# lspart on the real board, so it cannot be assumed.
if [ "$STATIC_CHUNKS" -gt 0 ]; then
    add "submission-model_mps_owncores" \
        "pair|static|||taskset -c $SUBMIT_CORE $RUNNER $MOB $DS --allocation whole --submit-threads 1|taskset -c $SUBMIT_CORE_B $RUNNER $RES $DS --allocation whole --submit-threads 1|75"
fi


# The declared-concurrency block. The subject is RetinaNet's aux8 build, the only
# engine in the corpus whose stream demand (9) exceeds the platform's default
# declaration (8), and therefore the only one where the work-queue half of a green
# context descriptor can bind. MobileNet is the co-runner: one stream, so it never
# competes for the declaration, and it is the network whose tail is best characterised.
#
# The three rows walk the ladder 8 -> 10, and then hold the number and change the
# scope, so each step varies exactly one thing.
add "workqueue-declaration_pair-sms-only" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --net $RN8:50:retinanet|"
add "workqueue-declaration_pair-device-ctx-10" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --wq-scope device_ctx --wq-limit 10 --net $MOB:50:mobilenet --net $RN8:50:retinanet|"
add "workqueue-declaration_pair-balanced-10" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --wq-scope green_ctx_balanced --wq-limit 10 --net $MOB:50:mobilenet --net $RN8:50:retinanet|"

# The gain attempt: three clients in one split, whose streams together ask for more
# concurrency than the platform declares by default (1 + 3 + 9 = 13 against 8). The
# split rounds up to 6 + 6 + 4 SMs, which is 16: percentages are rounded up to the
# 2-SM alignment, so 40/30/30 would ask for 20 SMs and be refused.
# declaration is worth anything, this is where it should show, because a gain is only
# possible where the default budget is exceeded.
add "workqueue-three-clients_three-clients-default" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:35:mobilenet --net $INC2:35:inception --net $RN8:25:retinanet|"
add "workqueue-three-clients_three-clients-balanced-13" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --wq-scope green_ctx_balanced --wq-limit 13 --net $MOB:35:mobilenet --net $INC2:35:inception --net $RN8:25:retinanet|"
add "workqueue-three-clients_three-clients-device-ctx-13" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --wq-scope device_ctx --wq-limit 13 --net $MOB:35:mobilenet --net $INC2:35:inception --net $RN8:25:retinanet|"

# Memory contention, block 1: the target's partition is FIXED at 8 SMs and only the
# co-runner's budget varies, so the three rows differ in the co-runner alone.
#
#   mobilenet_gc50 alone            exp1, already measured
#   mobilenet 50 + resnet 25        this arm: 8 + 4 SMs, four left idle
#   mobilenet 50 + resnet 50        exp2-1t 50_50, already measured
#
# The residual is then the target's concurrent latency over its isolated latency at the
# SAME SM count, and the question is whether it grows with the co-runner's budget.
add "memory-budget-greenctx_mobilenet8-resnet4" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --net $RES:25:resnet|"

# The interferer matrix. The target holds 8 SMs in every row — the same partition as the
# isolated reference and as the 8+4 and 8+8 rows of the budget arm — and the participant is
# a third client in the same joint split, doing one kind of work at one of two budgets.
#
#   mem_saxpy    256 MB buffers, three times its working set per pass: memory-bound, its
#                arithmetic negligible
#   compute_gemm 512 cubed fp32 with three 1 MB operands resident in L2: arithmetic-bound,
#                its DRAM traffic negligible
#
# The run's own log carries the participant's report — units, bytes, FLOPs and the rates
# they imply — so each row records not only what the target lost but how much load was
# actually applied to make it lose it.
add "memory-interferers_memsaxpy-4sm" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --interferer mem_saxpy:25|"
add "memory-interferers_memsaxpy-8sm" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --interferer mem_saxpy:50|"
add "memory-interferers_computegemm-4sm" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --interferer compute_gemm:25|"
add "memory-interferers_computegemm-8sm" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --interferer compute_gemm:50|"
# spin keeps one register-only kernel resident in its partition and touches no memory: what a
# co-runner costs by merely executing, with no traffic and almost no launches.
add "memory-interferers_spin-4sm" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --interferer spin:25|"
add "memory-interferers_spin-8sm" \
    "solo|off|||$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --interferer spin:50|"

# The same two kinds of co-runner, in a separate process under MPS static partitions.
#
# Partitions cover the whole device in two parts, so the split the daemon can express is
# 50/50: MobileNet confined to 8 SMs through its own partition and the co-runner to the
# other 8. Those are exactly the shares of the two green-context rows already measured
# (`mem_saxpy` at 8 SMs and `compute_gemm` at 8 SMs), so the process model is the only
# difference between the two sets of rows.
#
# The co-runner is utility/probes/itfproc: the same work as the in-process interferer, written as
# its own program, because a co-tenant is a process and not a thread. Same kernels, same
# buffer sizes, same reporting, so its own log states the load it applied.
#
# `mps_alone` is the reference that separates the cost of the mechanism from the cost of
# the co-runner: the same target, in the same 8-SM partition, with the other partition idle.
add "memory-mps-interferers_mps-alone" \
    "solo|static|||$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $MOB:50:mobilenet||50"
add "memory-mps-interferers_mps-memsaxpy-8sm" \
    "itf|static|||$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $MOB:50:mobilenet|$ROOT/utility/probes/itfproc mem_saxpy|50"
add "memory-mps-interferers_mps-computegemm-8sm" \
    "itf|static|||$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $MOB:50:mobilenet|$ROOT/utility/probes/itfproc compute_gemm|50"

# The fixed-budget measurement under MPS static partitions: MobileNet holds 8 SMs and ResNet
# is varied, the same two rows the green-context arm measures. A static partition set covers
# the device in two parts, so the 50:25 split is expressed as three partitions and the last
# quarter of the device is left unassigned, which is what the green-context arm does with its
# unused SMs. Both processes run their own submitter, as the frozen submission model requires.
add "memory-mps-budget_mps-mobilenet8-resnet4" \
    "pair|static|||$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $MOB:50:mobilenet|$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $RES:25:resnet|50:25"
add "memory-mps-budget_mps-mobilenet8-resnet8" \
    "pair|static|||$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $MOB:50:mobilenet|$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $RES:50:resnet|50:50"

# The 4-SM participants of the interferer matrix, under MPS static partitions. The ratio
# 50:25 creates three partitions - 8 SMs to the target, 4 to the participant, 4 unassigned -
# so the participant budget matches the green-context row it is paired with.
add "memory-mps-interferers-4sm_mps-memsaxpy-4sm" \
    "itf|static|||$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $MOB:50:mobilenet|$ROOT/utility/probes/itfproc mem_saxpy|50:25"
add "memory-mps-interferers-4sm_mps-computegemm-4sm" \
    "itf|static|||$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $MOB:50:mobilenet|$ROOT/utility/probes/itfproc compute_gemm|50:25"

# The two-thread check on the tail. One measured client with the interferer inside its own
# process, two submitting threads, both pinned. The tail of this arm says whether the pathology
# the submission model removed returns when a computing participant shares the process.
add "memory-two-threads_computegemm-8sm-two-threads" \
    "solo|off|||taskset -c $SUBMIT_CORE,$SUBMIT_CORE_B $RUNNER --submit-threads 2 --dataset $DS --allocation greenctx --net $MOB:50:mobilenet --interferer compute_gemm:50|"

# ---------------------------------------------------------------------------
# The work-queue descriptor, on engines that have streams to declare.
#
# RetinaNet aux8 runs nine streams (one main, eight auxiliary) and Inception aux2 three, so the
# descriptor's queue half is exercised here in a way it was not on the earlier pair, whose engines
# adopted almost none. Per client set: the SMs-only control, then the declared concurrency at the
# heaviest single client's stream count (9) and at the sum of all clients' streams, each under
# both sharing scopes. The declaration is a hint to the driver, not a cap, so a null result is a
# statement about this stack.
WQAUX="$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx"

add "workqueue-aux-streams_retinanet-retinanet-sms-only" \
    "solo|off|||$WQAUX --net $RN8:50:retinanet_a --net $RN8:50:retinanet_b|"
add "workqueue-aux-streams_retinanet-retinanet-device-ctx-9" \
    "solo|off|||$WQAUX --wq-scope device_ctx --wq-limit 9 --net $RN8:50:retinanet_a --net $RN8:50:retinanet_b|"
add "workqueue-aux-streams_retinanet-retinanet-balanced-9" \
    "solo|off|||$WQAUX --wq-scope green_ctx_balanced --wq-limit 9 --net $RN8:50:retinanet_a --net $RN8:50:retinanet_b|"
add "workqueue-aux-streams_retinanet-retinanet-device-ctx-18" \
    "solo|off|||$WQAUX --wq-scope device_ctx --wq-limit 18 --net $RN8:50:retinanet_a --net $RN8:50:retinanet_b|"
add "workqueue-aux-streams_retinanet-retinanet-balanced-18" \
    "solo|off|||$WQAUX --wq-scope green_ctx_balanced --wq-limit 18 --net $RN8:50:retinanet_a --net $RN8:50:retinanet_b|"

add "workqueue-aux-streams_retinanet-inception-sms-only" \
    "solo|off|||$WQAUX --net $RN8:50:retinanet --net $INC2:50:inception|"
add "workqueue-aux-streams_retinanet-inception-device-ctx-9" \
    "solo|off|||$WQAUX --wq-scope device_ctx --wq-limit 9 --net $RN8:50:retinanet --net $INC2:50:inception|"
add "workqueue-aux-streams_retinanet-inception-balanced-9" \
    "solo|off|||$WQAUX --wq-scope green_ctx_balanced --wq-limit 9 --net $RN8:50:retinanet --net $INC2:50:inception|"
add "workqueue-aux-streams_retinanet-inception-device-ctx-12" \
    "solo|off|||$WQAUX --wq-scope device_ctx --wq-limit 12 --net $RN8:50:retinanet --net $INC2:50:inception|"
add "workqueue-aux-streams_retinanet-inception-balanced-12" \
    "solo|off|||$WQAUX --wq-scope green_ctx_balanced --wq-limit 12 --net $RN8:50:retinanet --net $INC2:50:inception|"

add "workqueue-aux-streams_retinanet-retinanet-inception-sms-only" \
    "solo|off|||$WQAUX --net $RN8:35:retinanet_a --net $RN8:35:retinanet_b --net $INC2:25:inception|"
add "workqueue-aux-streams_retinanet-retinanet-inception-device-ctx-9" \
    "solo|off|||$WQAUX --wq-scope device_ctx --wq-limit 9 --net $RN8:35:retinanet_a --net $RN8:35:retinanet_b --net $INC2:25:inception|"
add "workqueue-aux-streams_retinanet-retinanet-inception-balanced-9" \
    "solo|off|||$WQAUX --wq-scope green_ctx_balanced --wq-limit 9 --net $RN8:35:retinanet_a --net $RN8:35:retinanet_b --net $INC2:25:inception|"
add "workqueue-aux-streams_retinanet-retinanet-inception-device-ctx-21" \
    "solo|off|||$WQAUX --wq-scope device_ctx --wq-limit 21 --net $RN8:35:retinanet_a --net $RN8:35:retinanet_b --net $INC2:25:inception|"
add "workqueue-aux-streams_retinanet-retinanet-inception-balanced-21" \
    "solo|off|||$WQAUX --wq-scope green_ctx_balanced --wq-limit 21 --net $RN8:35:retinanet_a --net $RN8:35:retinanet_b --net $INC2:25:inception|"

# ---------------------------------------------------------------------------
# Additivity: one target at 8 SMs (50 %), two co-runners at 4 SMs each (25 % + 25 %), so the
# device is fully used. The question is whether the interference a target takes from two
# co-runners is the sum of what each introduces alone.
#
# The triple {MobileNet, RetinaNet, Inception} is balanced: each is the target once, each pair is
# co-runners once. RetinaNet and Inception differ most of any pair available - 10 ms against
# 1.6 ms per inference, 480x640 against 299x299, ten outputs against one - which is what makes a
# sum worth reading. ResNet is a fourth target against the same pair, to compare with the 14.3 %
# and 25.9 % the corpus has for the 8/8 MobileNet-ResNet pair.
ADDT="$RUNNER --submit-threads 1 --dataset $DS --allocation greenctx"

add "interference-additivity_mobilenet-alone" \
    "solo|off|||$ADDT --net $MOB:50:mobilenet|"
add "interference-additivity_mobilenet+retinanet" \
    "solo|off|||$ADDT --net $MOB:50:mobilenet --net $RN:25:retinanet|"
add "interference-additivity_mobilenet+inception" \
    "solo|off|||$ADDT --net $MOB:50:mobilenet --net $INC:25:inception|"
add "interference-additivity_mobilenet+retinanet+inception" \
    "solo|off|||$ADDT --net $MOB:50:mobilenet --net $RN:25:retinanet --net $INC:25:inception|"

add "interference-additivity_retinanet-alone" \
    "solo|off|||$ADDT --net $RN:50:retinanet|"
add "interference-additivity_retinanet+mobilenet" \
    "solo|off|||$ADDT --net $RN:50:retinanet --net $MOB:25:mobilenet|"
add "interference-additivity_retinanet+inception" \
    "solo|off|||$ADDT --net $RN:50:retinanet --net $INC:25:inception|"
add "interference-additivity_retinanet+mobilenet+inception" \
    "solo|off|||$ADDT --net $RN:50:retinanet --net $MOB:25:mobilenet --net $INC:25:inception|"

add "interference-additivity_inception-alone" \
    "solo|off|||$ADDT --net $INC:50:inception|"
add "interference-additivity_inception+mobilenet" \
    "solo|off|||$ADDT --net $INC:50:inception --net $MOB:25:mobilenet|"
add "interference-additivity_inception+retinanet" \
    "solo|off|||$ADDT --net $INC:50:inception --net $RN:25:retinanet|"
add "interference-additivity_inception+mobilenet+retinanet" \
    "solo|off|||$ADDT --net $INC:50:inception --net $MOB:25:mobilenet --net $RN:25:retinanet|"

add "interference-additivity_resnet-alone" \
    "solo|off|||$ADDT --net $RES:50:resnet|"
add "interference-additivity_resnet+retinanet" \
    "solo|off|||$ADDT --net $RES:50:resnet --net $RN:25:retinanet|"
add "interference-additivity_resnet+inception" \
    "solo|off|||$ADDT --net $RES:50:resnet --net $INC:25:inception|"
add "interference-additivity_resnet+retinanet+inception" \
    "solo|off|||$ADDT --net $RES:50:resnet --net $RN:25:retinanet --net $INC:25:inception|"


# The stress model, under MPS static partitions only (MEASUREMENTS.md, "The stress model").
#
# stress-sensitivity: each target on 8 SMs, alone and beside the throttled SAXPY on the other 8 SMs
# at levels 0 ... 1, finer below 0.2 where the networks fall (itfproc saxpy --level, achieved
# bandwidth as a fraction of full speed).
# The target's slowdown against its alone row, per level, is its sensitivity curve.
#
# stress-validation: each target on 8 SMs beside each other network on 4 SMs, as a separate process.
# These pairs are the ground truth of the prediction and never enter the model: the model is built
# from the reporter (utility/reporter.sh) and from stress-sensitivity alone.
#
# A network co-runner runs long enough to outlast the target and is stopped by the sweep; its CSVs go
# to shared memory, never to the working directory.
STRESS_TARGETS="mobilenet:$MOB resnet:$RES inception:$INC retinanet:$RN"
stress_target() { echo "$RUNNER --submit-threads 1 --dataset $DS --allocation whole --net $2:50:$1"; }
for t in $STRESS_TARGETS; do
    tn="${t%%:*}"; te="${t#*:}"
    add "stress-sensitivity_${tn}-alone" "solo|static|||$(stress_target "$tn" "$te")||50"
    for p in 000 002 004 006 008 010 015 020 030 040 050 060 070 080 090 100; do
        add "stress-sensitivity_${tn}-saxpy-L$p" \
            "itf|static|||$(stress_target "$tn" "$te")|$ROOT/utility/probes/itfproc saxpy --level $((10#$p / 100)).$(printf '%02d' $((10#$p % 100)))|50:50"
    done
    # The same curves with the stressor on 4 SMs, the co-runner geometry of stress-validation
    # (target {0,1,4,5}, stressor {2,6}), and their own alone baseline.
    add "stress-sensitivity-4sm_${tn}-alone" "solo|static|||$(stress_target "$tn" "$te")||50"
    for p in 000 002 004 006 008 010 015 020 030 040 050 060 070 080 090 100; do
        add "stress-sensitivity-4sm_${tn}-saxpy-L$p" \
            "itf|static|||$(stress_target "$tn" "$te")|$ROOT/utility/probes/itfproc saxpy --level $((10#$p / 100)).$(printf '%02d' $((10#$p % 100)))|50:25"
    done
    # The baseline: the same target beside time-based SAXPY on 8 SMs (all threads on for D of every
    # 1 ms), D = 0 ... 1.0, and its own alone row.
    add "stress-sensitivity-tb_${tn}-alone" "solo|static|||$(stress_target "$tn" "$te")||50"
    add "stress-sensitivity-tb_${tn}-tb-D000" \
        "itf|static|||$(stress_target "$tn" "$te")|$ROOT/utility/probes/itfproc saxpy --level 0|50:50"
    for d in 010 020 030 040 050 060 070 080 090 100; do
        add "stress-sensitivity-tb_${tn}-tb-D$d" \
            "itf|static|||$(stress_target "$tn" "$te")|$ROOT/utility/probes/itfproc saxpy --period-us 1000 --duty $((10#$d / 100)).$(printf '%02d' $((10#$d % 100)))|50:50"
    done
    for c in $STRESS_TARGETS; do
        cn="${c%%:*}"; ce="${c#*:}"
        [ "$cn" = "$tn" ] && continue
        add "stress-validation_${tn}+${cn}" \
            "itf|static|||$(stress_target "$tn" "$te")|$RUNNER --submit-threads 1 --dataset $DS --fixed 32 --allocation whole --net $ce:100:$cn --min-inferences 100 --min-duration 600 --csv /dev/shm/fs-stress-co-$cn|50:25"
    done
done
