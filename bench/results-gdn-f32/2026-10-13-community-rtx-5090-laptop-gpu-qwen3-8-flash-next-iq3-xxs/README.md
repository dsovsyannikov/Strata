# Community benchmark on NVIDIA GeForce RTX 5090 Laptop GPU

Measured on 2026-10-13 with `tools/strata_bench.py` (suite 1, 1 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 Laptop GPU, 24 GiB VRAM
- CPU: AMD Ryzen AI 9 HX 370 w/ Radeon 890M (12 cores, 24 threads); installed RAM: 63.1 GiB; storage: not measured
- OS: Windows 11 (10.0.26300); GPU driver: 616.92; CUDA: not read
- Strata: project 0.1.43, commit 7695859a; engine 0.1.43
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack iq3_xxs --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 262144 --kv int8 --kv-resident 32768 --conversation-cache-mib 4096 --conversation-cache-slots 3 --session-min-free-mib 102400 --conversation-cache-storage disk --conversation-cache-dir disk-cache-test --mmap-experts --conversation-cache-min-free-mib 768
```

## Method

Run by `python tools/strata_bench.py --runs 1` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 1 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 1 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 1 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 3 | not measured | 10.3 (9.9-12.7) | 2.57 (2.35-2.65) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 1 | 229.9 (229.9-229.9) | not measured | 17.62 (17.62-17.62) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 1 | 325.5 (325.5-325.5) | not measured | 49.49 (49.49-49.49) |
| agent session, follow-up turns | 14468 (14468-14468) at the last turn | see below | up to 64 | 1 | see below | 11.7 (7.1-14.2) | 12.60 (9.37-13.55) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.8 % (68.8-68.8); earlier-turn tokens read again per session: 3 (3-3) (earlier turns that the cache did not hold; the 13634 (13634-13634) new tokens of the files are read once); first turn (nothing cached) first token: 7.41 s (7.41-7.41).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 83.9 | 7.413 | 10.0 |
| 1 | 3704 | 624 | 3080 | 247.1 | 12.621 | 14.2 |
| 2 | 5237 | 3767 | 1470 | 159.3 | 9.368 | 12.1 |
| 3 | 8681 | 5301 | 3380 | 252.1 | 13.55 | 10.9 |
| 4 | 11605 | 8744 | 2861 | 230.0 | 12.602 | 11.7 |
| 5 | 14468 | 11622 | 2846 | 229.3 | 12.578 | 7.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 218 s.
