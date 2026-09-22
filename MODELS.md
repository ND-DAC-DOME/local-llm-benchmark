# Model weights used

All weights come from Hugging Face. The snapshot is the repository commit that was served; re-runs should use the same one.

| Used in runs | Repository | Snapshot (commit) | File(s) | SHA-256 |
|---|---|---|---|---|
| 1, 7, 8 | `unsloth/Muse-Glimmer-30B-GGUF` | `faa5b025c584459c13febfa5c59883516710ae39` | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` (15.9 GB) | `82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38` |
| 7, 8 | `unsloth/Muse-Glimmer-30B-GGUF` | same | `dflash-kquant.gguf` (1.6 GB, DFlash drafter for llama.cpp) | `27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc` |
| 1, 6 | `unsloth/Qwen3.6-27B-GGUF` | `82d411acf4a06cfb8d9b073a5211bf410bfc29bf` | `Qwen3.6-27B-UD-Q4_K_XL.gguf` (17.6 GB) | `ff6941ded525b34eb159496762c29dd0ec6e71dc31b74d57e75d871a03eec259` |
| 2, 3, 8 | `unsloth/Qwen3.8-27B-NVFP4` | `f0b7c9e722f5565102fff8481c99e4d86ae099c7` | safetensors, 23.4 GB: NVFP4 MLPs (layers 0-55) 8.4 GB, FP8 MLPs (layers 56-63) 2.1 GB, FP8 attention + lm_head 8.5 GB, BF16 embeddings 2.5 + MTP head 0.85 + vision 0.9 | pinned by `--revision` |
| 5 (Spark) | `unsloth/Qwen3.8-27B-NVFP4` | `57926baca9a82b4d6906b43f2750d55315f5b10f` | same weights (only README commits between the two) | — |
| 4, 7 | `nvidia/Muse-Glimmer-30B-NVFP4` | `47818374517751c48c55cde2621594926b1888b6` | safetensors, 24.7 GB (ModelOpt AutoQuant: NVFP4/FP8/BF16) | pinned by `--revision` |
| 7 | `meta-models/Muse-Glimmer-30B-assistant` | `e8192f3a8f617f74be2ce220360c89ef4789f39f` | DFlash drafter for vLLM, 5.1 GB BF16 | pinned by `revision` in `--speculative-config` |

vLLM runs pin the snapshot on the command line. llama.cpp's `-hf repo:quant` cannot pin a commit, so check the GGUF files after download:

```bash
./verify_models.sh
```

## How the checkpoint mix was determined

The per-tensor dtypes quoted above (and the "bytes per decode pass" estimate in `results.md`) were read from the
safetensors headers, not from `config.json`:

```bash
python3 - <<'EOF'
import json, struct, glob, collections, re
S = glob.glob("$HF_CACHE/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/f0b7c9e*/")[0]
B = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}
tot = collections.defaultdict(lambda: collections.defaultdict(int))
for f in ["model.safetensors", "model_mtp.safetensors"]:
    with open(S + f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        size = B.get(v["dtype"], 1)
        for d in v["shape"]: size *= d
        g = ("mlp" if ".mlp." in k else "linear_attn" if "linear_attn" in k else "self_attn" if "self_attn" in k
             else "lm_head" if "lm_head" in k else "embed" if "embed_tokens" in k else "visual" if "visual" in k
             else "mtp" if f.startswith("model_mtp") else "other")
        tot[g][v["dtype"]] += size
for g, d in tot.items(): print(g, {k: round(x / 1e9, 2) for k, x in d.items()}, "GB")
EOF
```

A `weight_packed` tensor in `U8` with a `weight_scale` in `F8_E4M3` is an NVFP4 layer (4-bit values, block-16 FP8 scales);
a `weight` in `F8_E4M3` with a per-channel `weight_scale` is FP8. For this checkpoint: MLP layers 0-55 NVFP4, MLP layers
56-63 FP8, all attention projections and `lm_head` FP8, embeddings / MTP head / vision encoder BF16.
