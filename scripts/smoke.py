"""Exercise real OpenAI-compatible ordinary and SSE chat APIs (stdlib only)."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:48173")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "smoke.json")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    def get(path):
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return response.status, response.read().decode("utf-8")

    health_status, _ = get("/health")
    _, models_raw = get("/v1/models")
    model = json.loads(models_raw)["data"][0]["id"]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain why the sky is blue in two short sentences."}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 64,
    }
    result = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base,
        "health_status": health_status,
        "models": json.loads(models_raw),
        "request": payload,
    }

    def request(stream):
        return urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(dict(payload, stream=stream)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    start = time.perf_counter()
    with urllib.request.urlopen(request(False), timeout=180) as response:
        ordinary = json.loads(response.read())
        ordinary_status = response.status
    ordinary_seconds = time.perf_counter() - start
    if ordinary_status != 200 or not ordinary["choices"][0]["message"]["content"]:
        raise RuntimeError("Ordinary chat response was empty or unsuccessful")
    result["ordinary"] = {"status": ordinary_status, "elapsed_seconds": ordinary_seconds, "response": ordinary}

    start = time.perf_counter()
    sse_lines = []
    content = []
    first_content_seconds = None
    done = False
    with urllib.request.urlopen(request(True), timeout=180) as response:
        stream_status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            sse_lines.append(line)
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    if first_content_seconds is None:
                        first_content_seconds = time.perf_counter() - start
                    content.append(text)
    if stream_status != 200 or not done or not content:
        raise RuntimeError("SSE response did not contain content and a DONE marker")
    result["stream"] = {
        "status": stream_status,
        "elapsed_seconds": time.perf_counter() - start,
        "first_content_seconds": first_content_seconds,
        "content": "".join(content),
        "done": done,
        "raw_sse_lines": sse_lines,
    }
    _, metrics = get("/metrics")
    required = ["vllm:request_queue_time_seconds", "vllm:time_to_first_token_seconds", "vllm:e2e_request_latency_seconds"]
    if any(name not in metrics for name in required):
        raise RuntimeError("Required latency metrics are missing")
    result["metrics_verified"] = required
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_name("metrics-after-smoke.txt").write_text(metrics, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ordinary_seconds": ordinary_seconds, "stream_first_content_seconds": first_content_seconds, "response": ordinary["choices"][0]["message"]["content"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
