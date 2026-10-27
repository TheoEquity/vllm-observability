"""Run the unmodified vLLM v0.8.0 client; collect host/Prometheus evidence.

Run from this experiment after building scripts/benchmark-client.Dockerfile.
The finite concurrency test measures a saturated closed-loop client, not an
open-loop arrival-rate or service-level capacity guarantee.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "vllm-v0.8.0"
IMAGE = "vllm-observability-benchmark:v0.8.0"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TOKENIZER = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/" + REVISION
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = ROOT / "results" / "benchmark" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(url, payload=None, raw=False):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode()
    return text if raw else json.loads(text)


def client_command():
    return ["docker", "run", "--rm", "--network", "vllm-observability_default", "--entrypoint", "python3", "-e", "VLLM_USE_V1=0", "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HF_HUB_DISABLE_TELEMETRY=1", "-v", str(ROOT) + ":/experiment", "-v", "vllm-observability_huggingface-cache:/root/.cache/huggingface:ro", "-w", "/experiment/vendor/vllm-v0.8.0", IMAGE]


def counters(text):
    names = {"vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:request_success_total", "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count"}
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in names:
            values[name] = values.get(name, 0) + float(line.split()[-1])
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    print("OUTPUT " + str(OUT), flush=True)
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if line and not line.startswith("#"))
    api = "http://127.0.0.1:" + env.get("VLLM_PORT", "48173")
    prom = "http://127.0.0.1:" + env.get("PROMETHEUS_PORT", "48174")
    fetch(api + "/health", raw=True)
    assert fetch(api + "/v1/models")["data"][0]["id"] == MODEL
    manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8-sig"))
    for entry in manifest["files"]:
        assert hashlib.sha256((VENDOR / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], entry
    save(OUT / "client-image.json", json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{json .}}"], text=True)))
    (OUT / "client-packages.txt").write_text(subprocess.check_output(client_command() + ["-m", "pip", "freeze"], text=True), encoding="utf-8")
    save(OUT / "protocol.json", {"started_at_utc": utc(), "concurrencies": [1, 2, 4, 8, 16], "formal_rounds": 3, "minimum_requests": 200, "minimum_official_duration_seconds": 60, "warmup_requests_per_level": 10, "seed": 0, "target_input_tokens": 256, "requested_output_tokens": 256, "ignore_eos": True, "client_has_gpu_passthrough": False, "server_revision": REVISION, "notes": ["Each official invocation also sends one untimed initial test request.", "Envelope UTC includes client container startup, imports, tokenizer loading and initial test. Official duration alone is the performance denominator.", "Synthetic IDs are decoded to text then retokenized by the server; nominal input_lens=256 are not actual server token counts.", "Shared host services remain running. GPU samples describe the whole GPU.", "Prometheus range data is at 5-second steps. Rolling 1-minute windows can span adjacent phases."]})
    # Inspect one actual synthetic prompt through non-streamed API usage.
    sample_code = "import json,numpy as np; from transformers import AutoTokenizer; from benchmark_dataset import RandomDataset; np.random.seed(0); t=AutoTokenizer.from_pretrained('" + TOKENIZER + "'); r=RandomDataset().sample(tokenizer=t,num_requests=1,input_len=256,output_len=256,range_ratio=1,prefix_len=0)[0]; print(json.dumps({'prompt':r.prompt,'nominal_input_tokens':r.prompt_len,'retokenized_input_tokens':len(t(r.prompt,add_special_tokens=False).input_ids)}))"
    sample = json.loads(subprocess.check_output(client_command() + ["-c", sample_code], text=True, timeout=120).splitlines()[-1])
    payload = {"model": MODEL, "prompt": sample["prompt"], "temperature": 0, "max_tokens": 256, "ignore_eos": True}
    response = fetch(api + "/v1/completions", payload)
    save(OUT / "synthetic-usage-sample.json", {"at_utc": utc(), "sample": sample, "request": payload, "response": response, "scope": "One illustrative synthetic prompt; actual lengths vary across the full random workload."})
    assert response["usage"]["completion_tokens"] == 256
    print("SYNTHETIC_SAMPLE " + json.dumps({"nominal_input_tokens": 256, "usage": response["usage"]}), flush=True)
    stop = threading.Event()
    sampler_errors = []

    def sample_gpu():
        with (OUT / "gpu.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sampled_at_utc", "gpu_timestamp_local", "gpu_index", "gpu_name", "utilization_gpu_percent", "memory_used_mib", "memory_total_mib", "temperature_c", "power_draw_w"])
            while not stop.is_set():
                try:
                    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], text=True, timeout=8)
                    stamp = utc()
                    for row in csv.reader(rows.splitlines(), skipinitialspace=True):
                        writer.writerow([stamp] + row)
                    handle.flush()
                except Exception as exc:
                    sampler_errors.append({"at_utc": utc(), "error": str(exc)})
                stop.wait(2)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    phases = []
    experiment_start = time.time()

    def run_one(concurrency, count, kind, round_number):
        label = f"c{concurrency:02d}-{kind}-{round_number:02d}-n{count}"
        directory = OUT / label
        directory.mkdir()
        fetch(api + "/health", raw=True)
        before = fetch(api + "/metrics", raw=True)
        (directory / "metrics-before.txt").write_text(before, encoding="utf-8")
        command = client_command() + ["benchmark_serving.py", "--backend", "vllm", "--base-url", "http://vllm:8000", "--endpoint", "/v1/completions", "--model", MODEL, "--tokenizer", TOKENIZER, "--dataset-name", "random", "--random-input-len", "256", "--random-output-len", "256", "--random-range-ratio", "1", "--random-prefix-len", "0", "--request-rate", "inf", "--max-concurrency", str(concurrency), "--seed", "0", "--ignore-eos", "--num-prompts", str(count), "--disable-tqdm", "--save-result", "--save-detailed", "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,95,99", "--result-dir", "/experiment/" + directory.relative_to(ROOT).as_posix(), "--result-filename", "official.json"]
        container_name = "vllm-benchmark-" + label
        command[2:2] = ["--name", container_name]
        record = {"case_id": label, "kind": kind, "round": round_number, "concurrency": concurrency, "num_prompts": count, "started_at_utc": utc(), "command": command, "official_initial_test_requests": 1}
        save(directory / "phase.json", record)
        print(f"START {label} {record['started_at_utc']}", flush=True)
        try:
            with (directory / "client.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            record["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            record["returncode"] = "timeout-1200s"
            subprocess.run(["docker", "stop", "--time", "5", container_name], capture_output=True, timeout=20)
        record["finished_at_utc"] = utc()
        # V0 metric counters may be updated on the engine's periodic stats tick.
        time.sleep(6)
        after = fetch(api + "/metrics", raw=True)
        (directory / "metrics-after.txt").write_text(after, encoding="utf-8")
        record["metrics_after_at_utc"] = utc()
        a, b = counters(after), counters(before)
        record["server_counter_deltas_including_initial_test"] = {key: a[key] - b.get(key, 0) for key in a}
        if (directory / "official.json").exists():
            official = json.loads((directory / "official.json").read_text())
            record.update({key: official.get(key) for key in ["duration", "completed", "total_input_tokens", "total_output_tokens", "request_throughput", "output_throughput", "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "mean_e2el_ms", "p95_e2el_ms", "p99_e2el_ms"]})
            record["all_requests_successful"] = official["completed"] == count and not any(official["errors"])
            record["all_output_lengths_256"] = official["output_lens"] == [256] * count and official["total_output_tokens"] == count * 256
            record["formal_size_duration_pass"] = count >= 200 and official["duration"] >= 60
        save(directory / "phase.json", record)
        phases.append(record)
        save(OUT / "phases.json", phases)
        columns = ["case_id", "kind", "round", "concurrency", "num_prompts", "started_at_utc", "finished_at_utc", "duration", "completed", "total_input_tokens", "total_output_tokens", "request_throughput", "output_throughput", "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "mean_e2el_ms", "p95_e2el_ms", "p99_e2el_ms", "all_requests_successful", "all_output_lengths_256", "formal_size_duration_pass"]
        with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(phases)
        print(f"END {label} exit={record['returncode']} duration={record.get('duration')} output_tok_s={record.get('output_throughput')}", flush=True)
        assert record["returncode"] == 0 and record.get("all_requests_successful") and record.get("all_output_lengths_256"), record
        return record

    try:
        # Small official-client calibration before any full workload.
        run_one(1, 2, "calibration", 1)
        for concurrency in [1, 2, 4, 8, 16]:
            run_one(concurrency, 10, "warmup", 1)
            count = 200
            round_number = 1
            attempt = 1
            while round_number <= 3:
                # Classification is corrected to probe if its official duration is short.
                record = run_one(concurrency, count, "candidate", attempt)
                if record["formal_size_duration_pass"]:
                    record["kind"] = "formal"
                    record["formal_round"] = round_number
                    round_number += 1
                else:
                    record["kind"] = "probe"
                    count = max(count + 1, math.ceil(count * 70 / record["duration"]))
                save(OUT / record["case_id"] / "phase.json", record)
                save(OUT / "phases.json", phases)
                attempt += 1
        formal = [phase for phase in phases if phase["kind"] == "formal"]
        assert len(formal) == 15
        save(OUT / "verification.json", {"verified_at_utc": utc(), "formal_runs": len(formal), "formal_requests": sum(x["num_prompts"] for x in formal), "all_15_meet_minimum_requests_and_duration": all(x["formal_size_duration_pass"] for x in formal), "all_formal_requests_successful": all(x["all_requests_successful"] for x in formal), "all_formal_output_lengths_exactly_256": all(x["all_output_lengths_256"] for x in formal), "gpu_sampler_errors": sampler_errors})
    finally:
        stop.set()
        sampler.join(timeout=12)
        save(OUT / "gpu-sampler-errors.json", sampler_errors)
        end = time.time()
        # Preserve raw counters, gauges and histogram buckets, without interpolated summaries.
        query = {"query": '{job="vllm"}', "start": experiment_start, "end": end, "step": 5}
        save(OUT / "prometheus-range-request.json", query)
        save(OUT / "prometheus-range.json", fetch(prom + "/api/v1/query_range?" + urllib.parse.urlencode(query)))
        # Refresh CSV after candidate -> formal/probe classification.
        if phases:
            with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
                columns = list(phases[0]) + ["formal_round"]
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(phases)
        print("EVIDENCE " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
