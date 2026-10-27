"""Real Chinese cases and project-only restart evidence; stdlib only.

Run cases once, then restart with the printed evidence directory. Never retries
an answer to improve its content. Admin credentials stay only in memory.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if line and not line.startswith("#"))
API = "http://127.0.0.1:" + ENV.get("VLLM_PORT", "48173")
PROM = "http://127.0.0.1:" + ENV.get("PROMETHEUS_PORT", "48174")
GRAF = "http://127.0.0.1:" + ENV.get("GRAFANA_PORT", "48175")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REV = "7ae557604adf67be50417f59c2c2f167def9a775"


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch(url, payload=None, admin=False, raw=False):
    headers = {"Content-Type": "application/json"}
    if admin:
        headers["Authorization"] = "Basic " + base64.b64encode(("admin:" + ENV["GRAFANA_ADMIN_PASSWORD"]).encode()).decode()
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as response:
        result = response.read().decode()
    return result if raw else json.loads(result)


def prom_range(start, end):
    params = {"query": '{job="vllm"}', "start": start, "end": end, "step": "5s"}
    return {"request": params, "response": fetch(PROM + "/api/v1/query_range?" + urllib.parse.urlencode(params))}


def metrics(path):
    path.write_text(fetch(API + "/metrics", raw=True), encoding="utf-8")


def request(case, index, payload):
    result = {"case_id": case, "index": index, "started_at_utc": utc(), "request": payload}
    clock = time.perf_counter()
    try:
        result["response"] = fetch(API + "/v1/chat/completions", payload)
        result["http_status"] = 200
    except Exception as error:
        result["error"] = repr(error)
    result.update(finished_at_utc=utc(), client_elapsed_seconds=time.perf_counter() - clock)
    return result


def annotation(case):
    payload = {"dashboardUID": "vllm-observability", "time": int(datetime.fromisoformat(case["started_at_utc"]).timestamp()*1000), "timeEnd": int(datetime.fromisoformat(case["finished_at_utc"]).timestamp()*1000), "tags": ["acceptance", case["case_id"]], "text": case["case_id"] + ": exact Chinese prompt and unedited responses in results/acceptance/" + case["run_id"] + "/" + case["case_id"] + ".json"}
    return {"request": payload, "response": fetch(GRAF + "/api/annotations", payload, admin=True)}


def cases(out):
    out.mkdir(parents=True, exist_ok=False)
    print("OUTPUT " + str(out), flush=True)
    stop = threading.Event()
    gpu_errors = []

    def sample_gpu():
        with (out / "gpu.csv").open("w", encoding="utf-8") as stream:
            stream.write("utc,index,memory_used_MiB,memory_total_MiB,utilization_gpu_pct\n")
            while not stop.is_set():
                try:
                    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
                    for row in rows.splitlines():
                        stream.write(utc() + "," + row.replace(" ", "") + "\n")
                    stream.flush()
                except Exception as error:
                    gpu_errors.append({"utc": utc(), "error": repr(error)})
                stop.wait(2)

    sampler = threading.Thread(target=sample_gpu)
    sampler.start()
    begin = time.time() - 15
    try:
        metrics(out / "metrics-before-A.txt")
        payload_a = {"model": MODEL, "messages": [{"role": "user", "content": "请用中文列出 HTTP 状态码 200、404、500 各自的含义，每个状态码一句话。"}], "temperature": 0, "seed": 0, "max_tokens": 256}
        a = {"case_id": "case-A", "run_id": out.name, "started_at_utc": utc(), "concurrency": 1, "requests": [request("case-A", 0, payload_a)]}
        a["finished_at_utc"] = utc()
        save(out / "case-A.json", a)
        save(out / "annotation-A.json", annotation(a))
        metrics(out / "metrics-after-A.txt")
        print("A complete; retaining first answer; 65s separation for rolling window", flush=True)
        time.sleep(65)
        metrics(out / "metrics-before-B.txt")
        payload_b = {"model": MODEL, "messages": [{"role": "user", "content": "请用中文为一个由 React 前端、Python 后端和 SQLite 数据库组成的小型语言模型对比平台编写操作手册。按 1 到 10 编号，依次说明输入提示词、同时请求两个模型、等待结果、比较回答、投票、保存记录、查看排行榜、查看个人历史、处理模型错误、保护密钥。每一步写出用户操作和系统行为，最后解释为什么回答质量和生成速度需要分别评价。"}], "temperature": 0, "seed": 0, "max_tokens": 384}
        b = {"case_id": "case-B", "run_id": out.name, "started_at_utc": utc(), "concurrency": 16, "minimum_submission_seconds": 60, "requests": []}
        deadline = time.perf_counter() + 60

        def worker(index):
            rows = []
            sequence = 0
            while time.perf_counter() < deadline:
                rows.append(request("case-B", index + sequence*16, payload_b))
                sequence += 1
            return rows

        with ThreadPoolExecutor(max_workers=16) as pool:
            for rows in pool.map(worker, range(16)):
                b["requests"].extend(rows)
        b["requests"].sort(key=lambda r: (r["started_at_utc"], r["index"]))
        b["finished_at_utc"] = utc()
        save(out / "case-B.json", b)
        save(out / "annotation-B.json", annotation(b))
        print("B complete: " + str(len(b["requests"])) + " requests; waiting for final scrapes", flush=True)
        time.sleep(12)
        metrics(out / "metrics-after-B.txt")
        end = time.time()
        save(out / "prometheus-range.json", prom_range(begin, end))
        save(out / "dashboard.json", fetch(GRAF + "/api/dashboards/uid/vllm-observability"))
        save(out / "annotations.json", fetch(GRAF + "/api/annotations?dashboardUID=vllm-observability&limit=100"))
        save(out / "window.json", {"start": begin, "end": end, "timezone": "UTC"})
    finally:
        stop.set()
        sampler.join()
        save(out / "gpu-errors.json", gpu_errors)


def snapshot(out, label, window):
    ids = subprocess.check_output(["docker", "compose", "ps", "-q"], cwd=ROOT, text=True).split()
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
    safe = [{"Id": c["Id"], "Name": c["Name"], "Image": c["Image"], "StartedAt": c["State"]["StartedAt"], "Status": c["State"]["Status"], "Health": c["State"].get("Health", {}).get("Status"), "Mounts": c["Mounts"], "Cmd": c["Config"]["Cmd"]} for c in containers]
    volumes = sorted({m["Name"] for c in safe for m in c["Mounts"] if m["Type"] == "volume"})
    save(out / (label + "-containers.json"), safe)
    save(out / (label + "-volumes.json"), json.loads(subprocess.check_output(["docker", "volume", "inspect", *volumes], text=True)))
    cache_code = "import os,json,hashlib; p='/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/" + REV + "'; print(json.dumps([dict(name=n,realpath=os.path.realpath(p+'/'+n),size=os.stat(p+'/'+n).st_size,mtime_ns=os.stat(p+'/'+n).st_mtime_ns) for n in sorted(os.listdir(p))]))"
    cache = json.loads(subprocess.check_output(["docker", "compose", "exec", "-T", "vllm", "python3", "-c", cache_code], cwd=ROOT, text=True))
    data = {"at_utc": utc(), "cache": cache, "models": fetch(API + "/v1/models"), "history": prom_range(window["start"], window["end"]), "dashboard": fetch(GRAF + "/api/dashboards/uid/vllm-observability"), "annotations": fetch(GRAF + "/api/annotations?dashboardUID=vllm-observability&limit=100")}
    save(out / (label + ".json"), data)
    metrics(out / (label + "-metrics.txt"))
    return safe, data


def restart(out):
    window = json.loads((out / "window.json").read_text())
    before_c, before = snapshot(out, "restart-before", window)
    started = utc()
    proc = subprocess.run(["docker", "compose", "restart", "vllm", "prometheus", "grafana"], cwd=ROOT, text=True, capture_output=True)
    save(out / "restart-command.json", {"started_at_utc": started, "finished_at_utc": utc(), "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
    assert proc.returncode == 0
    deadline = time.monotonic() + 600
    attempts = []
    while time.monotonic() < deadline:
        try:
            fetch(API + "/health", raw=True)
            assert fetch(GRAF + "/api/health")["database"] == "ok"
            targets = fetch(PROM + "/api/v1/targets")
            assert any(t["labels"].get("job") == "vllm" and t["health"] == "up" for t in targets["data"]["activeTargets"])
            break
        except Exception as error:
            attempts.append({"utc": utc(), "error": repr(error)})
            print("Waiting for restarted project health: " + str(len(attempts)), flush=True)
            time.sleep(10)
    else:
        save(out / "restart-health-attempts.json", attempts)
        raise RuntimeError("Project health did not recover within 600s")
    save(out / "restart-health-attempts.json", attempts)
    after_c, after = snapshot(out, "restart-after", window)
    logs = subprocess.check_output(["docker", "compose", "logs", "--no-color", "--since", started, "vllm"], cwd=ROOT, text=True, encoding="utf-8", errors="replace")
    (out / "restart-vllm.log").write_text(logs, encoding="utf-8")
    verify_restart(out)


def verify_restart(out):
    before_c, after_c = (json.loads((out / (label + "-containers.json")).read_text(encoding="utf-8")) for label in ["restart-before", "restart-after"])
    before, after = (json.loads((out / (label + ".json")).read_text(encoding="utf-8")) for label in ["restart-before", "restart-after"])
    # Docker may reorder Mounts after restart; compare entries by destination.
    mount_sets = lambda containers: {c["Name"]: sorted(c["Mounts"], key=lambda m: m["Destination"]) for c in containers}
    check = {"verified_at_utc": utc(), "same_container_ids": [c["Id"] for c in before_c] == [c["Id"] for c in after_c], "same_mounts": mount_sets(before_c) == mount_sets(after_c), "all_started_at_changed": all(a["StartedAt"] != b["StartedAt"] for a, b in zip(before_c, after_c)), "same_cache_file_sizes_mtimes_targets": before["cache"] == after["cache"], "same_fixed_model": before["models"]["data"][0]["id"] == after["models"]["data"][0]["id"] == MODEL, "same_historical_prometheus_response": before["history"] == after["history"], "same_dashboard": before["dashboard"]["dashboard"] == after["dashboard"]["dashboard"], "same_annotations": before["annotations"] == after["annotations"], "healthy_api_grafana_prometheus_target": True}
    save(out / "restart-verification.json", check)
    print(json.dumps(check), flush=True)
    assert all(value is True for key, value in check.items() if key != "verified_at_utc"), check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cases", "restart", "verify-restart"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or ROOT / "results" / "acceptance" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    {"cases": cases, "restart": restart, "verify-restart": verify_restart}[args.mode](output)
