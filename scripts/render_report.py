"""Render an offline evidence report and standard matplotlib charts from raw files."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import statistics
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def seconds(text):
    return datetime.fromisoformat(text).timestamp()


def counters(path):
    data = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            name = line.split("{", 1)[0].split()[0]
            data[name] = data.get(name, 0) + float(line.split()[-1])
    return data


def main(acceptance, benchmark):
    acceptance, benchmark = acceptance.resolve(), benchmark.resolve()
    a, b = (load(acceptance / ("case-" + key + ".json")) for key in ["A", "B"])
    phases = load(benchmark / "phases.json")
    formal = [phase for phase in phases if phase["kind"] == "formal"]
    # Check primary official output, not just the upstream verification summary.
    official = [(phase, load(benchmark / phase["case_id"] / "official.json")) for phase in formal]
    assert len(official) == 15
    assert all(result["completed"] == phase["num_prompts"] and result["duration"] >= 60 and result["completed"] >= 200 and all(n == 256 for n in result["output_lens"]) and not any(result["errors"]) for phase, result in official)
    dashboard = load(acceptance / "dashboard.json")["dashboard"]
    query_at = seconds(b["finished_at_utc"]) + 10
    queried = []
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            query = {"query": target["expr"], "time": query_at}
            with urllib.request.urlopen("http://127.0.0.1:48174/api/v1/query?" + urllib.parse.urlencode(query)) as response:
                result = json.load(response)
            queried.append({"panel": panel["title"], "legend": target["legendFormat"], "request": query, "response": result})
    (acceptance / "panel-queries.json").write_text(json.dumps(queried, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = load(acceptance / "prometheus-range.json")["response"]["data"]["result"]
    def gauge_max(name, case):
        values = [float(value) for series in raw if series["metric"]["__name__"] == name for timestamp, value in series["values"] if seconds(case["started_at_utc"]) <= timestamp <= seconds(case["finished_at_utc"]) and math.isfinite(float(value))]
        return max(values) if values else None
    observations = {}
    for key, case in [("A", a), ("B", b)]:
        before, after = (counters(acceptance / f"metrics-{when}-{key}.txt") for when in ["before", "after"])
        queue_count = after["vllm:request_queue_time_seconds_count"] - before["vllm:request_queue_time_seconds_count"]
        queue_sum = after["vllm:request_queue_time_seconds_sum"] - before["vllm:request_queue_time_seconds_sum"]
        kv = gauge_max("vllm:gpu_cache_usage_perc", case)
        observations[key] = {"requests": len(case["requests"]), "successful_http": sum(r.get("http_status") == 200 for r in case["requests"]), "finish_reasons": dict(Counter(r["response"]["choices"][0]["finish_reason"] for r in case["requests"] if "response" in r)), "output_tokens": sum(r["response"]["usage"]["completion_tokens"] for r in case["requests"] if "response" in r), "duration_seconds": seconds(case["finished_at_utc"]) - seconds(case["started_at_utc"]), "client_median_seconds": statistics.median(r["client_elapsed_seconds"] for r in case["requests"]), "queue_observation_count": queue_count, "queue_mean_seconds": queue_sum/queue_count if queue_count else None, "sampled_max_running": gauge_max("vllm:num_requests_running", case), "sampled_max_waiting": gauge_max("vllm:num_requests_waiting", case), "sampled_max_kv_percent": None if kv is None else kv*100}
    (acceptance / "case-observations.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    table = []
    for level in [1, 2, 4, 8, 16]:
        runs = [result for phase, result in official if phase["concurrency"] == level]
        rates = [run["output_throughput"] for run in runs]
        ax.scatter([level]*3, rates, s=35, color="#1b65a5")
        ax.plot([level-.25, level+.25], [statistics.median(rates)]*2, color="#ca4c1c", linewidth=3)
        table.append([str(level), str(runs[0]["completed"]), f'{min(rates):.1f}–{max(rates):.1f}', f'{statistics.median(rates):.1f}', f'{statistics.median(run["p95_ttft_ms"] for run in runs):.1f}', f'{statistics.median(run["p95_e2el_ms"] for run in runs):.1f}'])
    ax.set(xticks=[1, 2, 4, 8, 16], xlabel="Closed-loop client concurrency", ylabel="Generated tokens / second", title="Official vLLM 0.8.0 benchmark: three formal runs per level")
    ax.grid(alpha=.2)
    fig.text(.5, .01, "256 target input / 256 actual output tokens; fixed seed; shared RTX 4060 Laptop GPU", ha="center", fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(acceptance / "benchmark-throughput.png"); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    for ax, directory, title, column, time_column in [(axes[0], benchmark, "Whole-device physical GPU memory: benchmark", "memory_used_mib", "sampled_at_utc"), (axes[1], acceptance, "Whole-device physical GPU memory: Chinese cases A / B", "memory_used_MiB", "utc")]:
        with (directory / "gpu.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        ax.plot([datetime.fromisoformat(row[time_column]) for row in rows], [float(row[column]) for row in rows], linewidth=1)
        ax.set(title=title, ylabel="MiB used (all workloads)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc)); ax.grid(alpha=.2)
        ax.set_xlabel("2026-10-27 UTC; host samples every ~2 seconds")
        if directory == acceptance:
            for case in [a, b]:
                ax.axvspan(datetime.fromisoformat(case["started_at_utc"]), datetime.fromisoformat(case["finished_at_utc"]), alpha=.12, label=case["case_id"])
            ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(acceptance / "gpu-memory.png"); plt.close(fig)

    md, web = [], []
    def heading(text, level=2):
        md.append("#"*level + " " + text); web.append(f"<h{level}>{html.escape(text)}</h{level}>")
    def para(text):
        md.append(text); web.append("<p>" + html.escape(text) + "</p>")
    def code(text):
        md.append("```text\n" + text + "\n```"); web.append("<pre>" + html.escape(text) + "</pre>")
    def link(path, label):
        relative = path.relative_to(ROOT).as_posix(); md.append(f"[{label}]({relative})"); web.append(f'<p><a href="{relative}">{html.escape(label)}</a></p>')
    def image(path, label):
        relative = path.relative_to(ROOT).as_posix(); md.append(f"![{label}]({relative})"); web.append(f'<figure><a href="{relative}"><img src="{relative}" alt="{html.escape(label)}"></a><figcaption>{html.escape(label)}；点击原图查看。</figcaption></figure>')
    def data_table(headers, rows):
        md.append("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows))
        web.append("<div class=table><table><thead><tr>" + "".join("<th>"+html.escape(x)+"</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>"+html.escape(x)+"</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table></div>")
    heading("vLLM 本地 GPU 实验：请求、回答与监控证据", 1)
    para("本页是由原始数据生成的离线证据报告。记录 2026-10-27 实测：本地 Qwen2.5-0.5B-Instruct 经 vLLM 生成回答，Prometheus 采样，Grafana 展示。API 与监控链路通过；回答有指令遵循和截断问题，不能据此声称模型质量合格。")
    para("固定环境：vLLM 0.8.0 V0，模型 revision " + "7ae557604adf67be50417f59c2c2f167def9a775" + "；FP16、上下文 2048、gpu-memory-utilization=0.5、max-num-seqs=16。宿主为共享 RTX 4060 Laptop GPU，其他原有工作负载保持运行。")
    link(ROOT / "compose.yaml", "固定部署配置")
    heading("首次中文请求与监控")
    para("A/B 使用不同 prompt、不同输出上限，是功能与可观测性演示；不能仅凭两例延迟差推断并发的因果效果。控制负载的量化比较见下方官方 15 轮。所有首次响应均保留，未为得到更好回答重试。")
    for key, case in [("A", a), ("B", b)]:
        obs = observations[key]; first = case["requests"][0]
        heading("Case " + key + ("：单次请求" if key == "A" else "：C16 持续请求"), 3)
        para(case["started_at_utc"] + " → " + case["finished_at_utc"] + "（UTC）")
        para("输入 prompt："); code(first["request"]["messages"][0]["content"])
        para("首次实际回答（原文）："); code(first["response"]["choices"][0]["message"]["content"])
        para(f'HTTP 200 数量 {obs["successful_http"]}/{obs["requests"]}；finish_reason={obs["finish_reasons"]}；总输出 {obs["output_tokens"]} tokens；阶段 {obs["duration_seconds"]:.3f}s；客户端完成延迟中位数 {obs["client_median_seconds"]:.3f}s。temperature=0、seed=0、max_tokens={first["request"]["max_tokens"]}；完整逐请求数据见链接。')
        gauge_text = "请求区间短于采样步长，区间内没有 range 评估点，不能得到 running/waiting/KV 的阶段峰值" if obs["sampled_max_kv_percent"] is None else f'running 最大值 {obs["sampled_max_running"]}、waiting 最大值 {obs["sampled_max_waiting"]}、KV 使用率最大值 {obs["sampled_max_kv_percent"]:.3f}%'
        gauge_limit = "未获得阶段峰值；队列均值来自请求前后计数差。" if obs["sampled_max_kv_percent"] is None else "这些 gauge 最大值仅代表采样观察，不是连续时间真实峰值；未观察到 waiting 不代表从未排队。"
        para(f'监控观测（5 秒步长）：{gauge_text}；原始队列 sum/count 差分均值 {obs["queue_mean_seconds"]*1000:.3f}ms（{obs["queue_observation_count"]:.0f} 次观察）。{gauge_limit}')
        para("人工观察：" + ("200/404/500 的核心含义大致对应，但每码多行，未遵守一句话要求；404 的‘不可达/无法访问’和 500 的‘请求被拒绝/需要重新尝试’表述过宽。不能从 500 单独推断重试会成功。" if key == "A" else "首例自行加入登录、账户管理等未要求内容，没有按指定操作顺序回答，随后进入 React 安装章节并在输出上限处截断，未给出最后的质量/速度解释。此处只检查直接可见的内容、格式与截断，未做全面质量评测。"))
        link(acceptance / ("case-" + key + ".json"), "完整 payload、所有原始响应、finish_reason、UTC 与客户端时间")
        image(acceptance / ("grafana-case-" + key + ".png"), "真实 Grafana Case " + key + "，绝对 UTC 时间窗、六面板及阶段 annotation")
    para("截图直接来自真实 Grafana。截图范围由对应请求起止前后留白构成，dashboard timezone=UTC，查询与截图绝对 URL 均已保存。无请求时的直方图分位数保留 NaN/空白，没有补成 0。A 是稀疏单请求，1 分钟 token rate 会被时间窗摊薄，不能与 182 tokens / 1.98s 的客户端阶段平均直接相比。")
    link(acceptance / "screenshots.json", "截图 UTC 范围与真实 dashboard URL")
    link(acceptance / "panel-queries.json", "B 结束后固定时间点六面板原始 PromQL 及响应")
    link(acceptance / "prometheus-range.json", "A/B 原始 Prometheus range 数据")
    link(acceptance / "case-observations.json", "A/B 统计与监控差分")

    heading("固定负载的并发比较")
    para("官方 vLLM v0.8.0 未修改脚本，固定 seed=0；目标输入 256 tokens、实际输出固定 256 tokens、ignore-eos、每档预热 10 请求。并发 1/2/4/8/16，各 3 轮，每轮至少 200 请求和 60 秒。已重新读取 15 份 official.json 核对：3,339 正式请求均成功，854,784 输出 tokens，所有输出长度为 256。")
    data_table(["并发", "每轮请求", "吞吐范围 tok/s", "吞吐中位数 tok/s", "各轮p95 TTFT中位数 ms", "各轮p95 E2E中位数 ms"], table)
    image(acceptance / "benchmark-throughput.png", "蓝点为每轮结果，红横线为三轮中位数；数据来自 official.json")
    para("在这组固定工作负载下，吞吐随并发增加。C16 吞吐仍有轮间波动；这些数据不证明服务已经达到最大容量。TTFT 与 E2E 是官方客户端样本统计，表中是各轮 p95 的中位数，未把三轮混合后重算 p95。")
    para("这是闭环饱和负载，客户端 semaphore 限制在途请求；TTFT 从取得客户端槽位后计时，不含等槽时间。正式 duration 作为吞吐分母；UTC envelope 还包含容器启动、导入、tokenizer 加载和初始测试，仅用于定位。每次官方调用另有 1 条不计正式的初始请求。C16 的 200 请求 / 44.744s 是 probe，正式三轮改为各 313 请求。")
    para("输入长度 256 是 synthetic 名义目标，文本 decode/retokenize 后服务端可能不同；保存的实际样例为 269 prompt tokens。官方 chunk 间隔不保证每 chunk 一个 token。本配置未开启 chunked prefill，连续批处理不证明 prefill/decode 混合批次。较早 185309Z 运行在依赖导入阶段失败，已进行 health/models 读取，但未发送生成请求、未进入正式负载；保留证据但未计入成功结果。")
    link(benchmark / "summary.csv", "全阶段 summary.csv")
    link(benchmark / "phases.json", "所有阶段与 official.json 路径")
    link(benchmark / "verification.json", "正式请求验证摘要")
    heading("宿主 GPU 显存")
    image(acceptance / "gpu-memory.png", "宿主 nvidia-smi 全设备物理显存时间曲线")
    para("官方压测 GPU CSV 观察到 6,531–6,975 MiB，全设备含其他程序；这不是 vLLM 进程独占显存。vLLM 预分配 KV 存储，物理显存与 KV 块使用百分比不同。CSV 约 2 秒采样，可能遗漏短峰值。")
    link(benchmark / "gpu.csv", "压测原始 GPU CSV")
    link(acceptance / "gpu.csv", "中文案例原始 GPU CSV")

    heading("三服务重启与持久性")
    restart = load(acceptance / "restart-verification.json")
    data_table(["检查", "实测结果"], [[key, str(value)] for key, value in restart.items()])
    para("请求结束后只执行本 compose project 的 vllm、prometheus、grafana restart；未停止其他项目。重启前后保存容器 ID/StartedAt、Mounts 和 volume inspect 的非密钥字段。固定 revision 缓存文件的真实路径、大小和 mtime 保持相同；重启日志显示模型加载成功。它证明复用现存 snapshot 并成功启动，不是离线网络封锁测试。Prometheus 旧 UTC 区间返回完全相同；Grafana dashboard 与 annotations 保持相同。重启后 API、Grafana 数据库及 Prometheus target 均恢复健康，三服务继续运行。")
    para("检查脚本初次逐数组比对 Mounts 时报告 false；原始文件显示只发生了 Grafana Mounts 顺序变化。改为按 Destination 排序比对后为 true，初次结果保留为 restart-initial-verification.json，没有再次重启。截图时普通 Chromium launcher 返回 spawn UNKNOWN，已安装的 Chromium headless shell 可启动并用于真实截图；普通 launcher 失败原因未独立确认。")
    link(acceptance / "restart-before.json", "重启前：缓存、旧范围、dashboard、annotations")
    link(acceptance / "restart-after.json", "重启后：相同检查")
    link(acceptance / "restart-before-containers.json", "重启前容器及挂载（未保存环境变量）")
    link(acceptance / "restart-after-containers.json", "重启后容器及挂载（未保存环境变量）")
    link(acceptance / "restart-verification.json", "重启逐项比对")
    link(acceptance / "restart-vllm.log", "重启 vLLM 原始日志")
    heading("结论的边界")
    para("Prometheus 每 5 秒采样；Grafana 的 rate/分位数使用滚动 1 分钟，临近阶段会混入同一窗口。直方图分位数是桶插值，队列首桶 0.3s；样本全在首桶时 p50≈0.15s、p95≈0.285s 不是真实排队延迟。低延迟队列优先看 sum/count 均值；该均值也不是分位数。")
    para("宿主共享、GPU CSV 含其他程序；观测并非生产 SLA 或开环可持续容量。电源状态只在压测中途检查一次，不能推断全程不变。早先 CUDA 分配失败与宿主 commit 压力/WSL 缓存有关的观察保留；drop_caches 后成功不能独立证明根因。未更改驱动、pagefile 或 WSL 配置。")
    para("已验证的是：GPU 生成/API、Prometheus/Grafana 链路、固定负载压测、真实截图关联、三服务重启与数据留存。未验证的是：通用回答质量、生产负载能力和首次 CUDA 分配失败根因。若继续，优先增加代表性任务与人工质量评测，再做可控到达率/更长持续时段测试；当前不增加这些范围。")
    (ROOT / "report.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    style = "body{font:16px/1.7 system-ui,Segoe UI,sans-serif;background:#f4f6f8;color:#172333;margin:0}main{max-width:1180px;margin:auto;padding:40px 28px;background:white}h1{font-size:30px}h2{margin-top:42px;border-bottom:2px solid #dbe4ed;padding-bottom:8px}h3{margin-top:32px}pre{background:#f0f4f8;border-left:4px solid #467aab;padding:18px;white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.65 system-ui}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:14px;color:#526173}a{color:#1c619d}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #d7e0e8;text-align:left}th{background:#eef3f8}.table{overflow:auto}"
    (ROOT / "report.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>vLLM 本地 GPU 证据报告</title><style>' + style + '</style><main>' + "\n".join(web) + '</main></html>', encoding="utf-8")
    print(json.dumps({"formal_runs_checked": len(official), "cases": observations, "report": "report.html"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    main(args.acceptance, args.benchmark)
