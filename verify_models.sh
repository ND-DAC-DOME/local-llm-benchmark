#!/usr/bin/env bash
# Check that the GGUF files in the local Hugging Face cache are byte-identical to the ones used for the published results.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$ROOT/config.env"; [ -f "$ROOT/config.local.env" ] && . "$ROOT/config.local.env"; set +a
rc=0
check() {  # check <repo dir> <file> <sha256>
  local f; f=$(ls "$HF_CACHE"/hub/"$1"/snapshots/*/"$2" 2>/dev/null | head -1)
  if [ -z "$f" ]; then echo "MISSING  $2 (not downloaded yet)"; rc=1; return; fi
  if [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$3" ]; then echo "OK       $2"; else echo "MISMATCH $2"; rc=1; fi
}
check models--unsloth--Muse-Glimmer-30B-GGUF dflash-kquant.gguf 27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc
check models--unsloth--Muse-Glimmer-30B-GGUF Muse-Glimmer-30B-UD-Q4_K_XL.gguf 82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38
check models--unsloth--Qwen3.6-27B-GGUF Qwen3.6-27B-UD-Q4_K_XL.gguf ff6941ded525b34eb159496762c29dd0ec6e71dc31b74d57e75d871a03eec259
exit $rc
