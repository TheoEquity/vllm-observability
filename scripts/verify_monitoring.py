"""Verify monitoring with one real request; save raw evidence (stdlib only)."""
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "monitoring"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if line and not line.startswith("#"))
    vllm = "http://127.0.0.1:" + env.get("VLLM_PORT", "48173")
    prom = "http://127.0.0.1:" + env.get("PROMETHEUS_PORT", "48174")
    grafana = "http://127.0.0.1:" + env.get("GRAFANA_PORT", "48175")
    auth = "Basic " + base64.b64encode(("admin:" + env["GRAFANA_ADMIN_PASSWORD"]).encode()).decode()

    def fetch(url, payload=None, admin=False, raw=False):
        headers = {"Content-Type": "application/json"}
        if admin:
            headers["Authorization"] = auth
        request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            text = response.read().decode()
        return text if raw else json.loads(text)

    def save(name, data):
        (OUT / name).write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Fresh installs need a pair of scrapes for rate().
    time.sleep(12)
    health = fetch(grafana + "/api/health")
    save("grafana-health.json", health)
    assert health["database"] == "ok", health
    targets = fetch(prom + "/api/v1/targets")
    save("prometheus-targets.json", targets)
    assert any(t["labels"].get("job") == "vllm" and t["health"] == "up" for t in targets["data"]["activeTargets"]), targets
    save("datasource.json", fetch(grafana + "/api/datasources/uid/vllm-prometheus", admin=True))
    ds_health = fetch(grafana + "/api/datasources/uid/vllm-prometheus/health", admin=True)
    save("datasource-health.json", ds_health)
    assert ds_health["status"] == "OK", ds_health
    dashboard = fetch(grafana + "/api/dashboards/uid/vllm-observability")
    save("dashboard-anonymous.json", dashboard)
    assert len(dashboard["dashboard"]["panels"]) == 6
    assert dashboard["meta"]["provisioned"] and not dashboard["meta"]["canEdit"]

    def query_all():
        return {str(panel["id"]): [{"legend": target["legendFormat"], "query": target["expr"], "response": fetch(prom + "/api/v1/query?" + urllib.parse.urlencode({"query": target["expr"]}))} for target in panel["targets"]] for panel in dashboard["dashboard"]["panels"]}

    save("queries-before.json", query_all())
    save("metrics-before.txt", fetch(vllm + "/metrics", raw=True))
    model = fetch(vllm + "/v1/models")["data"][0]["id"]
    payload = {"model": model, "messages": [{"role": "user", "content": "请用中文简要说明为什么比较语言模型时，需要同时记录回答质量、首字延迟和生成吞吐量。"}], "temperature": 0, "seed": 0, "max_tokens": 128}
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    response = fetch(vllm + "/v1/chat/completions", payload)
    elapsed = time.perf_counter() - clock
    ended = datetime.now(timezone.utc)
    assert response["choices"][0]["message"]["content"], response
    save("request-response.json", {"case_id": "monitoring-verification", "started_at_utc": started.isoformat(), "finished_at_utc": ended.isoformat(), "client_elapsed_seconds": elapsed, "request": payload, "response": response})
    time.sleep(12)
    save("metrics-after.txt", fetch(vllm + "/metrics", raw=True))
    queries = query_all()
    save("queries-after.json", queries)
    for panel, results in queries.items():
        for result in results:
            data = result["response"]
            assert data["status"] == "success" and data["data"]["result"], (panel, result)
            assert all(math.isfinite(float(series["value"][1])) for series in data["data"]["result"]), (panel, result)
    annotation = {"dashboardUID": "vllm-observability", "time": int(started.timestamp()*1000), "timeEnd": int(ended.timestamp()*1000), "tags": ["case", "monitoring-verification"], "text": "Single real Chinese prompt; exact request/response and UTC times: results/monitoring/request-response.json. Functional monitoring check, not a load benchmark."}
    save("annotation-request.json", annotation)
    save("annotation-response.json", fetch(grafana + "/api/annotations", annotation, admin=True))
    save("annotations.json", fetch(grafana + "/api/annotations?dashboardUID=vllm-observability"))
    summary = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "prometheus_target_up": True, "datasource_status": ds_health["status"], "anonymous_dashboard_panels": 6, "anonymous_can_edit": dashboard["meta"]["canEdit"], "all_six_panels_have_finite_samples": True, "request_completion_tokens": response["usage"]["completion_tokens"], "client_elapsed_seconds": elapsed, "note": "Running/waiting/KV are sampled gauges and may remain zero for this short request. Queue p50/p95 are coarse histogram estimates. This is not formal load testing."}
    save("verification.json", summary)
    print(json.dumps(summary))

if __name__ == "__main__":
    main()
