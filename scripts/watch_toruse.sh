#!/bin/bash
# Keep-alive watcher for the TorusE fix run (gamma=4).
# Behaviour every POLL_SECS:
#   1. If kg_toruse.log shows "Done"  -> TorusE finished, log result, exit.
#   2. Else if a run_kg_training process is alive -> training in progress, wait.
#   3. Else (not started / was killed) -> find a free GPU and (re)launch TorusE.
# This survives co-located jobs killing our run: it just relaunches on the next
# free card until a final result is produced. Logs to ~/KGOT/watcher_toruse.log.
#
# Settings are the ORIGINAL ones plus the TorusE gamma fix:
#   dim=256, 50 epochs, 32 negatives, gamma=4.0.

cd ~/KGOT
CAND_GPUS="0 1 2 3 4 5 6 7"
FREE_THRESH=1000       # MiB; a GPU below this is considered free
POLL_SECS=30
LOG=kg_toruse.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

log "TorusE keep-alive watcher started (dim=256, ep=50, nneg=32, gamma=4.0)."

while true; do
    # 1. Completed?
    if [ -f "$LOG" ] && grep -q "^Done" "$LOG"; then
        res=$(grep -E "Hits@" "$LOG" | tail -1)
        log "TorusE finished. Result: ${res}"
        log "Watcher exiting."
        break
    fi

    # 2. Currently training?
    if pgrep -f "run_kg_training" >/dev/null 2>&1; then
        gpu=$(cat .toruse_gpu 2>/dev/null)
        last=$(grep -E "Epoch|Hits@" "$LOG" 2>/dev/null | tail -1)
        log "Training in progress on GPU ${gpu}. last: ${last:-<starting>}"
        sleep "$POLL_SECS"
        continue
    fi

    # 3. Not running and not done -> find a free GPU and launch.
    free_gpu=""
    report=""
    for g in $CAND_GPUS; do
        u=$(mem_used "$g"); report="${report} g${g}=${u}"
        if [ -n "$u" ] && [ "$u" -lt "$FREE_THRESH" ] && [ -z "$free_gpu" ]; then
            free_gpu="$g"
        fi
    done

    if [ -n "$free_gpu" ]; then
        echo "$free_gpu" > .toruse_gpu
        log "Free GPU found: ${free_gpu} (${report} ). Launching TorusE."
        KG_GPU=$free_gpu KG_MODELS=toruse KG_TAG=toruse KG_EPOCHS=50 KG_NNEG=32 KG_GAMMA=4.0 \
            setsid bash scripts/launch_kg.sh > "$LOG" 2>&1 </dev/null &
        sleep 60   # give it time to come up before re-checking
    else
        log "No free GPU yet (${report} ). Waiting."
        sleep "$POLL_SECS"
    fi
done
