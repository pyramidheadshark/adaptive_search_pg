import asyncio
import time
import httpx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from tqdm.asyncio import tqdm

API_URL = "http://localhost:8000/api/v1/search"
CONCURRENCY_LEVELS = [1, 5, 10, 20]
REQUESTS_PER_LEVEL = 50
QUERY_EXAMPLE = "benefits of vitamin C for immune system"

async def measure_latency(client: httpx.AsyncClient, query: str):
    start_time = time.perf_counter()
    try:
        response = await client.post(API_URL, json={"query": query, "limit": 10})
        response.raise_for_status()
        data = response.json()
        total_time = (time.perf_counter() - start_time) * 1000  # ms
        return {
            "total_ms": total_time,
            "server_internal_ms": data.get("execution_time_ms", 0)
        }
    except Exception as e:
        return None

async def run_load_test(level: int):
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [measure_latency(client, QUERY_EXAMPLE) for _ in range(REQUESTS_PER_LEVEL)]
        results = await tqdm.gather(*tasks, desc=f"Level {level}", leave=False)
        return [r for r in results if r is not None]

async def main():
    all_results = []

    for level in CONCURRENCY_LEVELS:
        async with httpx.AsyncClient() as client:
            await measure_latency(client, "warmup")
        
        level_results = await run_load_test(level)
        for res in level_results:
            res["concurrency"] = level
            all_results.append(res)

    df = pd.DataFrame(all_results)
    
    df["network_overhead_ms"] = df["total_ms"] - df["server_internal_ms"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Распределение Latency (ms)", "Пропускная способность (RPS)"),
        horizontal_spacing=0.15
    )

    for level in CONCURRENCY_LEVELS:
        level_df = df[df["concurrency"] == level]
        fig.add_trace(
            go.Box(y=level_df["total_ms"], name=f"{level} User(s)", boxpoints='all', jitter=0.3),
            row=1, col=1
        )

    rps_data = []
    for level in CONCURRENCY_LEVELS:
        level_df = df[df["concurrency"] == level]
        total_time_sec = level_df["total_ms"].sum() / 1000 / level
        rps = level / (level_df["total_ms"].mean() / 1000)
        rps_data.append(rps)

    fig.add_trace(
        go.Bar(x=[f"{l} Users" for l in CONCURRENCY_LEVELS], y=rps_data, marker_color='rgb(55, 83, 109)'),
        row=1, col=2
    )

    fig.update_layout(
        title_text=f"Производительность Adaptive Search (N={len(df)} запросов)",
        template="plotly_white",
        showlegend=False,
        height=500,
        width=1100
    )
    
    fig.update_yaxes(title_text="Миллисекунды (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Запросов в секунду (RPS)", row=1, col=2)

    output_path = "data/performance_report.png"
    fig.write_image(output_path)
    fig.write_html("data/performance_report.html")

if __name__ == "__main__":
    asyncio.run(main())
