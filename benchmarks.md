## Benchmarks used in our local comparison

| Benchmark | What it measures | Dataset (the one we run) | Paper | Code |
|---|---|---|---|---|
| GSM8K | Grade-school math word problems (1,319 test items) | https://huggingface.co/datasets/openai/gsm8k | https://arxiv.org/abs/2110.14168 | https://github.com/openai/grade-school-math |
| MMLU-Pro | 10-option multiple choice across 14 subjects (~12k items) | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro | https://arxiv.org/abs/2406.01574 | https://github.com/TIGER-AI-Lab/MMLU-Pro |
| HumanEval | 164 Python functions, graded by running unit tests | https://huggingface.co/datasets/openai/openai_humaneval | https://arxiv.org/abs/2107.03374 | https://github.com/openai/human-eval |
| GPQA Diamond | 198 graduate-level science questions, 4 options | https://huggingface.co/datasets/hendrydong/gpqa_diamond_mc (ungated mirror) | https://arxiv.org/abs/2311.12022 | https://github.com/idavidrein/gpqa |
| LiveCodeBench v6 | 175 competitive-programming problems (contests Jan–Apr 2025), graded by running all tests | https://huggingface.co/datasets/livecodebench/code_generation_lite (`test6.jsonl`) | https://arxiv.org/abs/2403.07974 | https://github.com/LiveCodeBench/LiveCodeBench |

**Papers**

- GSM8K: "Training Verifiers to Solve Math Word Problems" (OpenAI, 2021).
- MMLU-Pro: "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark" (TIGER-Lab, 2024).
- HumanEval: "Evaluating Large Language Models Trained on Code" (OpenAI, 2021). This is the Codex paper; HumanEval was introduced in it.
- GPQA: "GPQA: A Graduate-Level Google-Proof Q&A Benchmark" (2023).
- LiveCodeBench: "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code" (2024).

**How we run them**

- GSM8K and MMLU-Pro: a fixed random sample of 200 items from the test split (seed 1234), the same items for every model. HumanEval: all 164 problems.
- Zero-shot, one chat turn per item. The model is asked to end with `Final answer: <number>` (GSM8K), `Answer: <letter>` (MMLU-Pro), or a single Python code block (HumanEval).
- GSM8K and MMLU-Pro are graded by exact match against the gold answer. HumanEval is graded by actually executing the problem's unit tests against the model's code; a problem counts only if every test passes.
- A response that hits the token limit (12,288) before giving an answer counts as wrong. The limit is the same for every model.
- Same sampling for every model: temperature 1.0, top_p 0.95 (the recommendation in both vendors' model cards); top_k is each vendor's own recommendation.
- Both models are 4-bit quantized (unsloth `UD-Q4_K_XL` GGUF), served with llama.cpp.
- Every prompt, response and verdict is saved to `results/<run>/<task>.jsonl` so scores can be audited.

**Caveats**

- Our prompts and answer parsers are our own, so these scores are not directly comparable to public leaderboards (which use different prompt formats, few-shot examples, etc.). They are meant for comparing models against each other under identical conditions.
- With 200 items the 95% margin of error is roughly ±3 to ±5 percentage points, depending on the score. Differences inside that margin should be read as a tie.
- GSM8K and HumanEval are close to saturated for current models of this size, so they work mainly as sanity checks. MMLU-Pro has more room to separate models.
- None of these is an agentic software-engineering test. For SWE-bench numbers published by the vendors, see `swe-bench-cards.md`.

Links checked on 2026-09-17: all resolve, and the arXiv titles match.
