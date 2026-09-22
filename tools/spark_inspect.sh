#!/usr/bin/env bash
# Read-only inspection of the DGX Spark's vLLM server, over the network. Every Spark number in results.md
# comes from one of these commands (or from the two prefix-cache tools next to this file). Nothing is changed
# on the Spark. Usage: tools/spark_inspect.sh [http://<spark-host>:18300]
set -uo pipefail
U=${1:-${SPARK_URL:-http://localhost:18300}}; U=${U%/v1}
M=$(curl -s -m 10 "$U/metrics") || { echo "cannot reach $U/metrics"; exit 1; }
g() { echo "$M" | grep -E "^vllm:$1\{" | head -1 | awk '{print $NF}'; }
echo "== served model";           curl -s -m 10 "$U/v1/models" | python3 -c "import sys,json; [print(' ',m['id'],'->',m.get('root')) for m in json.load(sys.stdin)['data']]"
echo "== load right now";         echo "  running=$(g num_requests_running) waiting=$(g num_requests_waiting) kv_cache_used=$(g kv_cache_usage_perc)"
echo "== speculative decoding (MTP) since server start"
d=$(g spec_decode_num_drafts_total); dt=$(g spec_decode_num_draft_tokens_total); ac=$(g spec_decode_num_accepted_tokens_total)
echo "  draft steps=$d drafted tokens=$dt accepted=$ac"; python3 -c "d,dt,ac=$d,$dt,$ac; print(f'  acceptance per drafted token {ac/dt:.3f} | tokens emitted per model pass {1+ac/d:.2f}')" 2>/dev/null
echo "$M" | grep -E "^vllm:spec_decode_num_accepted_tokens_per_pos_total" | sed 's/.*position="\([0-9]*\)".*} /  accepted at draft position \1: /'
echo "== prefix cache since server start (global counters; other users included)"
q=$(g prefix_cache_queries_total); h=$(g prefix_cache_hits_total); echo "  queried=$q hits=$h"; python3 -c "q,h=$q,$h; print(f'  hit rate {100*h/q:.1f}%')" 2>/dev/null
echo "== totals"; echo "  prompt tokens=$(g prompt_tokens_total) generated tokens=$(g generation_tokens_total)"
cat <<'TXT'
== on the Spark itself (needs a shell there; all read-only):
  docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'           # the sparkrun container
  ps -eo pid,user,args | grep -E 'vllm serve' | grep -v grep          # exact serve arguments (compare with docs/spark-*.yaml)
  nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv   # unified memory used by vLLM
  free -g                                                             # memory left for a second model
  docker exec <sparkrun container> sh -c 'grep -iE "NvFp4LinearKernel|Marlin|native support for FP4|attention backend|Initializing a V1" /tmp/sparkrun_serve.log'
                                                                      # which NVFP4 kernel and vLLM version are in use
  ss -tn state established '( sport = :18300 )'                       # who is connected right now
TXT
