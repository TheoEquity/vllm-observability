# Local vLLM observability experiment

This experiment is isolated from the comparison app. It runs a real local Qwen model on the NVIDIA GPU through Docker Desktop's WSL 2 backend.

Prerequisites: Docker Desktop with Linux containers and GPU support, an NVIDIA CUDA-compatible Windows driver, Python 3.10+ for the smoke script, internet access for the first download, and enough Docker data-disk space. No Hugging Face token is required for this public model.

From this directory:

```powershell
Copy-Item .env.example .env
# Replace the example password without printing the generated secret:
python -c "from pathlib import Path; import secrets; p=Path('.env'); p.write_text(p.read_text().replace('replace-with-a-random-local-password', secrets.token_urlsafe(32)))"
docker compose up -d vllm
docker compose logs -f vllm
# Once /health is ready:
python scripts/smoke.py
```

The API listens at `http://127.0.0.1:48173`; metrics are at `/metrics`. `.env.example` documents port overrides and the required local Grafana admin password; `.env` is ignored by Git. The named `huggingface-cache` volume persists downloaded weights in Docker's data disk. Stopping the service keeps its cache; no existing Docker data is removed.

## Fixed versions and choices

- Official image `vllm/vllm-openai:v0.8.0` pinned by digest `sha256:ce29db4955f06749d951cfa0ffd973741dcf9f1957d3166b6b5d75e86ff554ea`.
- Model `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16, maximum context 2048, GPU utilization budget 0.5, maximum 16 sequences.
- `--disable-frontend-multiprocessing` runs the V0 API and engine together, avoiding the additional frontend/engine process pair on this memory-constrained shared host. This let the second startup get past the first attempt's weight-allocation failure, although that second attempt still failed later during KV-cache allocation.
- `VLLM_USE_V1=0` deliberately selects this version's V0 engine so the experiment uses the inspected `vllm/engine/metrics.py` implementation, including queue time. This is a version and metric reproducibility choice. The image's [official Dockerfile](https://github.com/vllm-project/vllm/blob/v0.8.0/Dockerfile) uses CUDA 12.4.1; the local Windows driver 560.94 exceeds [CUDA 12.4 Update 1's Windows driver requirement 551.78](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html#id5).
- The [versioned metrics source](https://github.com/vllm-project/vllm/blob/v0.8.0/vllm/engine/metrics.py) exposes `vllm:request_queue_time_seconds`, TTFT, per-output-token time, end-to-end latency, tokens, running/waiting requests, and KV cache usage. Runtime metric names must still be checked against the saved scrape.
- The registry reported 8,853,450,085 bytes of compressed image layers; the model weight file is 988,097,824 bytes. Before pulling, the Docker data disk was on D: with approximately 69.64 GiB free. GPU pass-through was checked using the already-present `python:3.12-slim` image, avoiding an extra CUDA test-image download.

## Validation evidence

`scripts/smoke.py` calls `/health`, `/v1/models`, ordinary chat completion, streamed chat completion, and `/metrics`. It verifies nonempty responses and the SSE `[DONE]` marker, then saves the actual request, responses, UTC timestamps, client timings, and raw metrics under `results/`. Client first-content latency includes HTTP overhead and is distinct from the engine's TTFT histogram.

On 2026-10-27, the smoke script exited 0: health and both chat modes returned HTTP 200, the SSE stream ended with `[DONE]`, and the saved scrape contained queue time, TTFT, and end-to-end latency histograms. The ordinary request used 41 input tokens and produced the configured maximum 64 output tokens in 1.038 seconds; the stream completed in 0.762 seconds with first content at 0.089 seconds. These are two small functional requests, not a load benchmark. The sky explanation contains factual mistakes and ended with `finish_reason=length`; this check proves API transport and generation, not answer accuracy or instruction compliance. See `results/smoke.json` and `results/metrics-after-smoke.txt` for exact requests, responses, and timings. Monitoring, the official load sweep and screenshot/restart acceptance were subsequently executed; see the evidence report below.

The first two startups raised CUDA allocation errors despite reporting several GiB of free GPU memory. A standalone allocation test using the same PyTorch/CUDA image allocated 1.5 GiB successfully. Host memory pressure was observed, and WSL reported 23,469 MiB of buffer/cache after the image download. A one-time `sync` plus Linux `drop_caches=3` reduced buffer/cache to 812 MiB; the unchanged second-attempt configuration then started successfully. Immediate Windows committed-memory readings were 94% before and 95% after reclaim, so these observations do not independently prove the underlying allocation failure's cause. Later healthy-run committed memory was 86%. No driver, pagefile, WSL configuration, or pre-existing service was changed. The raw three startup logs, allocation test, and cache/memory measurements are in `results/deployment/`.

The healthy startup reported 0.93 GiB of model weights, a 4.00 GiB vLLM budget, 2.92 GiB reserved for KV cache, and successful CUDA graph capture. The subsequent host GPU reading was 6,523 MiB used across all workloads, with 1,435 MiB free; D: had 67.05 GiB free. These are observations from this run, not minimum resource guarantees.

The host also runs pre-existing services that Docker Desktop restored according to their restart policies. `results/deployment/shared-host.txt` records the observed containers and available RAM. Performance measurements on this machine include shared host activity.

## Prometheus and Grafana

After the GPU model is healthy, start only the monitoring services:

```powershell
docker compose up -d --no-deps prometheus grafana
# Functional validation: one real Chinese prompt, not a load benchmark.
python scripts/verify_monitoring.py
```

Prometheus: <http://127.0.0.1:48174>. Grafana dashboard: <http://127.0.0.1:48175/d/vllm-observability>. Both bind only to localhost. Anonymous access has the read-only Viewer role; the `admin` password comes from the ignored local `.env`. Keep the password private. Grafana stores its initial admin password in its persistent database; editing `.env` later does not reset an existing admin account.

The fixed linux/amd64 images are `prom/prometheus:v3.15.1@sha256:d445d1806b38f85b5c394a1b87f94113fde0473b45245922c2e8b1bd41fa68ae` and `grafana/grafana:12.4.14@sha256:1b59d2076a5e08e0e9929a1225f37869ee7719f4a11c004fbfcfa8b772858e89`. Their registry-reported compressed sizes were approximately 109 and 284 MiB. Named volumes `prometheus-data` and `grafana-data` preserve metrics, Grafana state, and annotations across container recreation. Configuration and dashboard JSON are read-only bind mounts under this experiment. Dashboard provisioning polls every 30 seconds to support Windows bind mounts. See [Prometheus Docker storage](https://prometheus.io/docs/prometheus/latest/installation/) and [Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/).

The six panels use names checked against the actual V0 `results/metrics-after-smoke.txt` response: TTFT p50/p95, generated tokens/s, running/waiting requests, queue p50/p95 plus mean, GPU KV cache percentage, and engine end-to-end latency p50/p95. Prometheus scrapes every 5 seconds; Grafana refreshes every 5 seconds. Throughput, latency estimates, and queue mean use a fixed rolling 1-minute window. Short gauge peaks can occur between scrapes, and adjacent cases less than a minute apart can share a window. These are server engine measurements; client first-content and HTTP completion times are separate measurements.

Histogram p50/p95 values are interpolated estimates, not exact request percentiles. In this version queue time's first bucket ends at 0.3 seconds. If all observations fall there, the estimated p50/p95 become 0.15/0.285 seconds even when actual queue times are much shorter. Use the mean calculated from histogram sum/count to assess those low queue times; its value is not a percentile. With zero observations, quantiles and mean remain NaN / no data; the dashboard never fills them with zero. See [Prometheus histogram_quantile behavior](https://prometheus.io/docs/prometheus/latest/querying/functions/#histogram_quantile). GPU KV cache percentage measures allocated KV blocks in use, not physical GPU memory. Host `nvidia-smi` CSV evidence is required for physical memory claims.

`verify_monitoring.py` checks the Prometheus target, datasource health, anonymous provisioned dashboard, and all six panels' real query results after one request. It saves exact Chinese prompt/response, client time, UTC timestamps, raw before/after scrapes, Prometheus API results, and a dashboard annotation in `results/monitoring/`. It reads the admin password without printing or saving it. This establishes the monitoring path; it does not establish load capacity, answer accuracy, or exact histogram percentiles.

Cases can be correlated using the dashboard's absolute UTC time range and Grafana annotations. An admin API call to `POST /api/annotations` accepts `dashboardUID: "vllm-observability"`, epoch-millisecond `time` / `timeEnd`, `tags`, and descriptive `text`; the verification script contains a working example with its request saved as evidence. Read-only viewers can see annotations. No extra plugin or UI is needed.

To stop only this experiment while preserving all named volumes:

```powershell
docker compose stop grafana prometheus vllm
```

Do not use `down -v` when retaining evidence or cached weights.
## Official serving benchmark

Build the short-lived CPU client, then run the fixed protocol from this directory:

```powershell
docker build -f scripts/benchmark-client.Dockerfile -t vllm-observability-benchmark:v0.8.0 .
python scripts/benchmark.py
```

The client uses the same pinned vLLM image plus `pandas==2.2.3`, `datasets==3.3.2` and a compatibility pin `pyarrow==19.0.1`, which the upstream dataset module imports even for synthetic data. It receives no GPU device. The server container is not modified. The exact built client image ID and installed packages are saved with every run. The four official scripts in `vendor/vllm-v0.8.0/` are unchanged copies of the [v0.8.0 benchmark sources](https://github.com/vllm-project/vllm/tree/v0.8.0/benchmarks), accompanied by the upstream Apache 2.0 license and a source-URL/SHA256 manifest checked before execution. The tokenizer is loaded offline from the exact model revision's mounted snapshot, because this benchmark version has no tokenizer-revision option.

The fixed synthetic workload targets 256 input and 256 output tokens with `--ignore-eos`, seed 0, no common prefix, and range ratio 1. Each concurrency level 1, 2, 4, 8, 16 gets ten warmup requests and three qualifying measured runs, each at least 200 requests and at least 60 seconds of official benchmark duration. An invocation below 60 seconds is retained as a probe and its count is increased for the next attempt. Every upstream invocation additionally makes one initial test request outside its measured interval. A two-request official calibration and one ordinary completion with saved API usage precede the workload. Output lengths are checked individually against server-reported completion usage: all successful requests must produce exactly 256 tokens. A failed request, incorrect length or stage exceeding 1,200 seconds stops the workload and preserves evidence.

This is a saturated closed-loop concurrency sweep with `--request-rate inf`, using a CPU client container on the same Docker network as vLLM. It excludes the Windows client network path. The client semaphore caps in-flight HTTP requests; per-request TTFT starts after the semaphore is acquired and therefore excludes time waiting for a client slot. A maximum of 16 client requests does not prove a fixed batch size of 16 or a nonzero server queue. These measurements do not establish a user-facing SLA or sustainable open-loop arrival rate.

`results/benchmark/<UTC-run-id>/` contains the protocol, per-phase command and process-envelope UTC timestamps, unchanged official detailed JSON and log, final `summary.csv`/`phases.json`, before/after raw metric scrapes, a full raw Prometheus range response, and host GPU samples every two seconds. Official `duration` is the performance denominator. The UTC envelope also includes container startup, Python imports, tokenizer loading and the initial test; use it to locate phases in Grafana, not to recompute token throughput. The final phase `kind` in `phase.json`, `phases.json` and CSV is authoritative; candidate directory names are retained after classification as `formal` or `probe`.

Interpretation limits:

- Synthetic token IDs are decoded into text and then retokenized by the server. Official `input_lens` and total input counts are nominal targets, not measured service input lengths. `synthetic-usage-sample.json` demonstrates actual API usage for one example; full-workload actual input totals are separately observable through server-counter deltas, which also include that invocation's initial test and can be affected by periodic statistics updates.
- The completions client requests streamed usage and uses `completion_tokens` for output counts. Its ITL samples represent stream-chunk intervals; a chunk need not correspond to one token. TPOT divides post-first-token latency by actual output count minus one. Official request end-to-end latency ends at the final token chunk, excluding the trailing usage/DONE delivery.
- This fixed V0 engine with context 2048 leaves chunked prefill disabled by default. Increased concurrency/continuous batching does not demonstrate a prefill-and-decode mixed batch. No scheduler configuration is changed for this sweep.
- Prometheus raw samples are saved at five-second resolution. The dashboard's rolling one-minute rates and histograms may overlap adjacent phases; select an interior phase window and identify this limitation in screenshots. Engine histogram quantiles remain bucket estimates and differ from the official client's exact sample percentiles.
- GPU CSV is whole-device physical memory/utilization on a shared host. Two-second sampling can miss short peaks. vLLM preallocates KV storage; physical memory usage is not equivalent to the fraction of KV blocks used. Existing host services remain running.


### Executed sweep: 2026-10-27

Run `results/benchmark/20261027T185626Z/` finished with exit code 0. All 15 formal runs passed: three per concurrency level, each at least 200 requests and 60 seconds. All 3,339 formal requests succeeded and each generated exactly 256 tokens (854,784 output tokens total). The final `summary.csv` records formal round numbers and classifications alongside the original detailed JSON. These are observations for this fixed synthetic workload on this shared machine.

| Client concurrency | Requests per formal run | Official duration range, seconds | Median of three output throughputs, token/s |
| --- | --- | --- | --- |
| 1 | 200 | 518.26–556.41 | 93.179 |
| 2 | 200 | 268.17–293.39 | 190.575 |
| 4 | 200 | 137.69–144.76 | 364.538 |
| 8 | 200 | 73.06–75.78 | 700.550 |
| 16 | 313 | 64.69–74.50 | 1222.555 |

The first concurrency-16 invocation used 200 requests and lasted only 44.744 seconds. It remains a probe; the three formal runs used 313 requests each. The single actual-input example had 269 prompt tokens for a nominal target of 256. Every recorded per-phase server request-success counter increment matched the measured request count plus the one upstream initial test; these counts are separate from formal benchmark results.

There are 1,889 GPU observations with no sampler errors, a median sampling interval of 2.106 seconds and a maximum interval of 2.384 seconds. Observed whole-device physical memory ranged from 6,531 to 6,975 MiB, including other host activity. The saved raw Prometheus range contains 249 time series. `client-device-requests.json` records `HostConfig.DeviceRequests=null` for an actual running benchmark container. `power-mid-run.json` records AC online, a full battery and the balanced power scheme at 19:14:57 UTC only; no start-of-run power state was captured, so full-run constancy is unknown.

The earlier `20261027T185309Z` directory is a preparation failure before generation requests: unpinned PyArrow 26 was incompatible with the image's NumPy 1.26.4. The script had already read `/health` and `/v1/models`, but did not enter the formal load workload. The client-only PyArrow 19.0.1 pin resolved that import failure. Its artifacts are retained as failed preparation evidence and excluded from the successful run. No server or monitoring configuration changed during benchmarking.

## Screenshot and restart acceptance

Open the [Chinese offline evidence report](report.html) directly in a browser, or read [report.md](report.md). The report is generated from raw measurements. It includes exact first Chinese prompts/responses, case UTC windows, six-panel real Grafana screenshots, benchmark charts, whole-device GPU memory curves, raw evidence links and limitations.

`results/acceptance/20261027T200700Z/` contains case A (one HTTP-state-code prompt, 182 output tokens, `stop`) and case B (16 concurrent clients submitting for at least 60 seconds, 240 successful requests, all 384-token `length` finishes). Their tasks/output limits differ, so A/B latency differences are not a controlled concurrency comparison. Both retain instruction-following issues rather than retrying for better answers. Case B's sampled running maximum was 16 and waiting maximum 0; this does not prove no transient queue. Queue sum/count mean was 27.908 ms, distinct from the coarse histogram's 285 ms p95 interpolation.

After these cases, only this project's three services were restarted. Raw before/after evidence confirms unchanged volume mounts, fixed-revision cache file targets/sizes/mtimes, identical historical Prometheus range results, dashboard and annotations; API/Grafana/Prometheus target recovered healthy. vLLM process counters reset as expected; historical Prometheus samples remained. The initial mount-array equality check failed because Docker reordered entries; normalized comparison passed, with the initial result retained. This restart test preserves the existing containers; it does not test deleting/recreating containers or a network-blocked cold start. All three services were left running.

To reproduce the case collection and project-only restart from this directory:

```powershell
python scripts/acceptance.py cases
# Use the output directory printed above (do not rerun cases merely to improve answers):
python scripts/acceptance.py restart --out results/acceptance/<UTC-run-id>
```

Report rendering requires Python's `matplotlib` in addition to the standard library. The screenshot helper uses Node plus Codex's bundled Playwright under the user's runtime directory and the installed Chromium headless shell 1248 under LocalAppData, both resolved from environment variables. Ordinary Chromium launcher startup returned `spawn UNKNOWN` on this host; the installed headless shell rendered the real Grafana successfully with `--disable-gpu`. No browser download or model configuration change was required.

```powershell
python scripts/render_report.py --acceptance results/acceptance/20261027T200700Z --benchmark results/benchmark/20261027T185626Z
node scripts/screenshots.cjs results/acceptance/20261027T200700Z
# Refresh only the report preview while preserving the existing Grafana screenshots:
node scripts/screenshots.cjs results/acceptance/20261027T200700Z --preview
```

The HTML uses relative local links; keep the experiment directory and its results together. Credentials are read only from the ignored `.env` and are never embedded in the report or inspect evidence.
