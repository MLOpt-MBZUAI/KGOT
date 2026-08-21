#!/bin/bash
# Watcher: restart the 5 KG training jobs on GPUs 3-7 once they are ALL free,
# WITHOUT preempting the user's temporary task.
#
# Because the GPUs are free right now (our jobs were just stopped) but the user's
# task has not started yet, the watcher first waits until it OBSERVES the user's
# task occupying the GPUs (Phase 1), and only then arms the "wait until all free"
# trigger (Phase 2). This guarantees we do not jump in before the user's task.
#
# Original training settings are preserved: dim=256, 50 epochs, 32 negatives,
# one model per GPU (rotate->3, toruse->4, complex_ff->5, pairre->6, mure->7).
#
# Runs persistently in the background; logs to ~/KGOT/watcher.log.

cd ~/KGOT
GPUS="3 4 5 6 7"
MEM_THRESH=1000        # MiB; a GPU below this is considered free
BUSY_THRESH=2000       # MiB; a GPU above this counts as "in use" by the other task
POLL_SECS=30
STABLE_PASSES=3        # consecutive free polls required (~90s) before launching
ARM_PASSES=1           # polls of observed busy-ness needed to arm

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

log "Watcher started. Phase 1: waiting to detect the other task on GPUs ${GPUS} (>${BUSY_THRESH} MiB)."

# --- Phase 1: wait until the user's task occupies at least one of GPUs 3-7 ---
armed=0
while [ "$armed" -lt "$ARM_PASSES" ]; do
    busy=0
    report=""
    for g in $GPUS; do
        u=$(mem_used "$g"); report="${report} g${g}=${u}"
        if [ -n "$u" ] && [ "$u" -ge "$BUSY_THRESH" ]; then busy=1; fi
    done
    if [ "$busy" -eq 1 ]; then
        armed=$((armed + 1))
        log "Detected other task running (${report} ). armed=${armed}/${ARM_PASSES}"
    fi
    sleep "$POLL_SECS"
done

log "Armed. Phase 2: waiting for GPUs ${GPUS} to all be free (<${MEM_THRESH} MiB)."

# --- Phase 2: wait until ALL GPUs 3-7 are free, stable for several polls ---
stable=0
while true; do
    all_free=1
    report=""
    for g in $GPUS; do
        u=$(mem_used "$g"); report="${report} g${g}=${u}"
        if [ -z "$u" ] || [ "$u" -ge "$MEM_THRESH" ]; then all_free=0; fi
    done
    if [ "$all_free" -eq 1 ]; then
        stable=$((stable + 1))
        log "All free (${report} ) stable=${stable}/${STABLE_PASSES}"
        [ "$stable" -ge "$STABLE_PASSES" ] && { log "GPUs 3-7 confirmed free. Launching."; break; }
    else
        [ "$stable" -ne 0 ] && log "Reset: not all free (${report} )"
        stable=0
    fi
    sleep "$POLL_SECS"
done

# --- Launch the 5 jobs with the ORIGINAL settings (unchanged) ---
launch() {
    local gpu=$1 model=$2
    KG_GPU=$gpu KG_MODELS=$model KG_TAG=$model KG_EPOCHS=50 KG_NNEG=32 \
        setsid bash scripts/launch_kg.sh > "kg_${model}.log" 2>&1 </dev/null &
    log "Launched ${model} on GPU ${gpu} (KG_DIM=256 KG_EPOCHS=50 KG_NNEG=32)"
}

launch 3 rotate
launch 4 toruse
launch 5 complex_ff
launch 6 pairre
launch 7 mure

sleep 5
log "All 5 training jobs launched. Watcher exiting."
