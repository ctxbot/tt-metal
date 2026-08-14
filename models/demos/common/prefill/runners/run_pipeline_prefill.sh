#!/usr/bin/env bash
# Thin launcher for the pipeline-parallel prefill runner under tt-run.
#
# Baseline config (mesh, layer split, chunk count, transport, PCC, PREFILL_* env) lives in the
# rank-binding YAML's global_env. In addition, this script injects every PREFILL_* var exported in
# the launching shell directly into the YAML's global_env on the fly (written to a temp binding) —
# ttrun itself only auto-propagates TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_ prefixes, so without this,
# shell PREFILL_* reach at most the rank co-located with mpirun via fork inheritance, skewing rank 0
# vs the rest. Shell exports OVERRIDE same-named global_env keys in the merged temp binding; the
# on-disk YAML is never modified.
#
# Usage:
#   run_pipeline_prefill.sh <rank_binding.yaml> [host_list] [tcp_iface]
#
#   <rank_binding.yaml>  path (relative to TT_METAL_HOME or absolute) to the tt-run rank binding.
#   [host_list]          mpirun --host value. Default: bh-glx-d03u02:1,bh-glx-d03u08:1 (2 galaxies).
#                        For a single-rank/one-galaxy binding pass e.g. bh-glx-d03u02:1.
#   [tcp_iface]          NIC for MPI TCP. Default: ens5f0np0 (the 10.32.24.x cluster net here).
#
# This launcher is model-agnostic (it only execs the common prefill_runner entry point). The
# rank-binding YAMLs + mesh-graph descriptors are topology config (model-agnostic; the model is
# selected by the binding's PREFILL_MANIFEST) and live at
# models/demos/common/prefill/runners/topology_configuration/. Pass your binding as $1.
#
# Examples:
#   # 2-galaxy D2D pipeline (connected MGD, FABRIC_2D):
#   ./run_pipeline_prefill.sh models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_rank_binding_2rank_d2d.yaml bh-glx-d07u02:1,bh-glx-d07u08:1
#
#   # 4-galaxy D2D pipeline (ring-chain host order — see the 4-galaxy connected MGD):
#   ./run_pipeline_prefill.sh models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_rank_binding_4rank_d2d.yaml bh-glx-d07u02:1,bh-glx-d07u08:1,bh-glx-d08u08:1,bh-glx-d08u02:1
#
#   # single-galaxy 1-rank full-model de-risk:
#   ./run_pipeline_prefill.sh models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_real_1galaxy_1rank.yaml bh-glx-d07u02:1
set -euo pipefail

RANK_BINDING="${1:?usage: run_pipeline_prefill.sh <rank_binding.yaml> [host_list] [tcp_iface]}"
HOST_LIST="${2:-bh-glx-d03u02:1,bh-glx-d03u08:1}"
TCP_IFACE="${3:-ens5f0np0}"

# TT_METAL_HOME = the tt-metal tree this script lives in
# (models/demos/common/prefill/runners -> 5 levels up).
TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME"
# Per-host LOCAL JIT cache. A shared (NFS) TT_METAL_CACHE makes both hosts write the same generated
# kernel files (defines_generated.h, ...) concurrently on a cold cache -> "Stale file handle" compile
# failures. /tmp is per-host, so each rank compiles into its own dir. ttrun auto-propagates TT_* vars.
export TT_METAL_CACHE="${PP_TT_METAL_CACHE:-/tmp/tt-metal-cache-pp}"
cd "$TT_METAL_HOME"

# -x PATH/LD_LIBRARY_PATH: ttrun only forwards TT_*/ARCH_*/... prefixed vars, not PATH, so peer ranks
# would otherwise resolve a bare `python3` to the system interpreter (no ttnn). Forwarding the launch
# host's PATH works only because every host's venv sits at the identical clone path.
#
# Merge shell-exported PREFILL_* into the binding's global_env (see header). Writes a temp copy of
# the binding next to it (same dir, so a relative mesh_graph_desc_path still resolves) and points
# ttrun at the copy. Requires python3 with PyYAML on the launch host (the tt-metal venv has it).
# NOTE: keep the gate assignment-based — `env | grep -q` under pipefail returns 141 (grep -q closes
# the pipe early, env dies on SIGPIPE), which would silently skip the merge.
PREFILL_PAIRS="$(env | grep -E '^PREFILL_[A-Za-z0-9_]+=' || true)"
if [ -n "$PREFILL_PAIRS" ]; then
  MERGED_BINDING="$(mktemp "$(dirname "$RANK_BINDING")/.rank_binding_merged.XXXXXX.yaml")"
  trap 'rm -f "$MERGED_BINDING"' EXIT
  PREFILL_PAIRS="$PREFILL_PAIRS" \
  python3 - "$RANK_BINDING" "$MERGED_BINDING" <<'EOF'
import os, re, sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
# PREFILL_PAIRS (newline-joined KEY=VALUE, pre-filtered) is the same set the bash gate tested.
shell = dict(line.split("=", 1) for line in os.environ["PREFILL_PAIRS"].splitlines() if line)
with open(src) as f:
    doc = yaml.safe_load(f)
env = doc.setdefault("global_env", {})
overrides = []
for k, v in shell.items():
    if k in env and str(env[k]) != v:
        overrides.append(f"{k}: {env[k]} -> {v}")
    env[k] = v
with open(dst, "w") as f:
    yaml.safe_dump(doc, f, sort_keys=False)
print(f"[run_pipeline_prefill] injected {len(shell)} shell PREFILL_* into global_env of {dst}")
for line in overrides:
    print(f"[run_pipeline_prefill]   shell overrides binding: {line}")
EOF
  RANK_BINDING="$MERGED_BINDING"   # point ttrun at the merged copy (plain assignment, not a per-command prefix)
fi

exec python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface "$TCP_IFACE" \
  --rank-binding "$RANK_BINDING" \
  --mpi-args "--host ${HOST_LIST} --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH" \
  python3 -m models.demos.common.prefill.runners.prefill_runne
