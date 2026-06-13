#!/usr/bin/env bash
# Continuous multi-shard driver for the PartVerse **XL** TRELLIS.2 pipeline.
#
# Sibling of run_all_shards_trellis2.sh, specialized for the raw-XL source:
#   * defaults to the XL config (data.source: partversexl_raw — NO pack)
#   * pins each shard to its FROZEN roster
#       configs/partversexl_shards/shard_<NN>.txt
#     (so the object set is stable + auditable, independent of what is currently
#      on disk under meshes/textured_part_glbs). Falls back to runtime --all
#      enumeration if a roster file is missing (set ROSTER_DIR="" to force --all).
#
# Runs each shard end-to-end (encode → VLM gen+gate_a → FLUX → 3D edit → Gate-E
# → del_add) BACK-TO-BACK, one at a time, on the same GPU/port pool. It never
# starts two shards concurrently, and before each shard it WAITS until no other
# run_pipeline_v3_shard_trellis2.sh worker is alive — safe to launch alongside a
# manual run; it clears the field first.
#
# Guarantees (identical to run_all_shards_trellis2.sh):
#   * Sequential — only one shard pipeline runs at any moment.
#   * Resumable — a completed shard drops <output>/_shard_<NN>.DONE; re-launching
#     skips DONE shards. The per-stage pipeline is itself resume-safe.
#   * Fail-safe — a non-zero shard drops <output>/_shard_<NN>.FAIL and the driver
#     STOPS (CONTINUE_ON_FAIL=1 to skip and keep going).
#   * Single-instance — a flock on <output>/_driver.lock prevents two drivers.
#
# Usage (launch in its own tmux so it survives disconnects):
#   tmux new-session -d -s xlall \
#     'bash run_all_xl_shards_trellis2.sh 2>&1 | tee data/Pxform_v2/partversexl_posthoc_no2dqc/_run_allshards.log'
#
# Env overrides (all optional):
#   SHARDS="01 02 ... 09"   Shards to run, in order (default 00..09)
#   CONFIG=<yaml>           Pipeline config (default XL raw-source config)
#   ROSTER_DIR=<dir>        Frozen-roster dir (default configs/partversexl_shards;
#                           empty string → use runtime --all enumeration instead)
#   GPUS="0,1,...,7"        GPU pool
#   CONCURRENCY=4           S1_PER_GPU_CONCURRENCY (VLM batching)
#   MACHINE_ENV=<path>      Machine env file
#   CONTINUE_ON_FAIL=0|1    Skip a failed shard instead of stopping (default 0)
#   FORCE_RERUN=0|1         Ignore .DONE sentinels and re-run every shard
#   POLL_SECS=30            How often to poll while waiting for the pool to free

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SHARDS="${SHARDS:-00 01 02 03 04 05 06 07 08 09}"
CONFIG="${CONFIG:-configs/pipeline_v3_trellis2_partversexl_posthoc_no2dqc.yaml}"
ROSTER_DIR="${ROSTER_DIR-configs/partversexl_shards}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CONCURRENCY="${CONCURRENCY:-4}"
MACHINE_ENV="${MACHINE_ENV:-configs/machine/local_trellis2.env}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
POLL_SECS="${POLL_SECS:-30}"

[ -f "$CONFIG" ] || { echo "[FATAL] config not found: $CONFIG"; exit 1; }

# Resolve the output tree from the config so sentinels/logs live with the data.
OUTPUT="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["data"]["output_dir"])' "$CONFIG" 2>/dev/null)"
[ -n "$OUTPUT" ] || { echo "[FATAL] could not resolve data.output_dir from $CONFIG"; exit 1; }
mkdir -p "$OUTPUT"

LOCK="$OUTPUT/_driver.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[FATAL] another driver already holds $LOCK — refusing to start a second one."
    exit 1
fi

ts() { date '+%F %T'; }
log() { echo "[$(ts)] [xl-driver] $*"; }

# True iff a REAL shard pipeline worker is alive (same matcher as the prod
# driver): match the run-script name, exclude tmux server / tee pipe / grep /
# this driver itself.
pool_busy() {
    pgrep -af 'run_pipeline_v3_shard_trellis2\.sh' 2>/dev/null \
        | grep -vE 'tmux|[[:space:]]tee([[:space:]]|$)|run_all_(xl_)?shards_trellis2|[[:space:]]grep([[:space:]]|$)' \
        | grep -q .
}

wait_for_pool_free() {
    local first=1
    while pool_busy; do
        if [ "$first" = 1 ]; then
            log "another shard pipeline is active — waiting for it to finish before starting the next shard…"
            first=0
        fi
        sleep "$POLL_SECS"
    done
    [ "$first" = 0 ] && log "pool is free — proceeding."
    return 0
}

log "=== continuous XL shard driver ==="
log "shards     : $SHARDS"
log "config     : $CONFIG"
log "roster dir : ${ROSTER_DIR:-<none → runtime --all>}"
log "gpus       : $GPUS   concurrency=$CONCURRENCY"
log "output     : $OUTPUT"
log "on-fail    : $([ "$CONTINUE_ON_FAIL" = 1 ] && echo 'skip & continue' || echo 'STOP')"
log "force-rerun: $FORCE_RERUN"

declare -a DONE_SHARDS=() FAIL_SHARDS=() SKIP_SHARDS=()

for SH in $SHARDS; do
    DONE_MARK="$OUTPUT/_shard_${SH}.DONE"
    FAIL_MARK="$OUTPUT/_shard_${SH}.FAIL"

    if [ "$FORCE_RERUN" != 1 ] && [ -f "$DONE_MARK" ]; then
        log "shard $SH: already DONE ($(cat "$DONE_MARK")) — skipping."
        SKIP_SHARDS+=("$SH")
        continue
    fi

    # Pin to the frozen roster when present; else fall back to runtime --all.
    OBJ_ENV=()
    if [ -n "$ROSTER_DIR" ]; then
        ROSTER="$ROSTER_DIR/shard_${SH}.txt"
        if [ -f "$ROSTER" ]; then
            OBJ_ENV=(OBJ_IDS_FILE="$ROSTER")
            log "shard $SH: roster $ROSTER ($(wc -l < "$ROSTER") objects)"
        else
            log "shard $SH: WARN no roster $ROSTER — using runtime --all enumeration."
        fi
    fi

    wait_for_pool_free

    rm -f "$FAIL_MARK"
    SHARD_LOG="$OUTPUT/_run_shard${SH}.log"
    log "shard $SH: STARTING → $SHARD_LOG"
    SECONDS=0

    # NOTE: OBJ_ENV is injected via `env` — array-expanded "VAR=val" words are NOT
    # recognized as bash assignment-prefixes (only literal tokens are), so passing
    # them straight before the command would try to exec the filename. `env` parses
    # its KEY=VAL args at runtime, so this works whether OBJ_ENV is set or empty.
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    S1_PER_GPU_CONCURRENCY="$CONCURRENCY" \
    SHARD="$SH" MACHINE_ENV="$MACHINE_ENV" PIPELINE_GPUS="$GPUS" \
        env "${OBJ_ENV[@]}" \
        bash run_pipeline_v3_shard_trellis2.sh "xl_shard${SH}" "$CONFIG" \
        2>&1 | tee "$SHARD_LOG"
    rc=${PIPESTATUS[0]}

    elapsed=$SECONDS
    hms=$(printf '%dh%02dm%02ds' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)))

    if [ "$rc" = 0 ]; then
        echo "$(ts) ok wall=${hms}" > "$DONE_MARK"
        log "shard $SH: DONE in $hms"
        DONE_SHARDS+=("$SH")
    else
        echo "$(ts) rc=$rc wall=${hms}" > "$FAIL_MARK"
        log "shard $SH: FAILED rc=$rc after $hms (see $SHARD_LOG)"
        FAIL_SHARDS+=("$SH")
        if [ "$CONTINUE_ON_FAIL" != 1 ]; then
            log "stopping (CONTINUE_ON_FAIL=0). Fix the issue and re-launch — DONE shards will be skipped."
            break
        fi
        log "CONTINUE_ON_FAIL=1 → moving on to the next shard."
    fi
done

log "=== summary ==="
log "done   : ${DONE_SHARDS[*]:-<none>}"
log "skipped: ${SKIP_SHARDS[*]:-<none>}"
log "failed : ${FAIL_SHARDS[*]:-<none>}"
[ ${#FAIL_SHARDS[@]} -eq 0 ]
