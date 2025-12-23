import asyncio
import time
import httpx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = "http://localhost:8000/api/v1/search"
CONFIG = {
    1: 1000,
    2: 1000,
    4: 1000,
    8: 1000
}

async def measure(client, semaphore, query):
    async with semaphore:
        start = time.perf_counter()
        try:
            resp = await client.post(API_URL, json={"query": query, "limit": 10}, timeout=20.0)
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed if resp.status_code == 200 else None
        except:
            return None

async def run_test():
    all_data = []
    async with httpx.AsyncClient() as client:
        await client.post(API_URL, json={"query": "warmup", "limit": 1})
        
        for level, count in CONFIG.items():
            print(f"Test: {level} User(s), {count} requests...")
            semaphore = asyncio.Semaphore(level)
            
            start_level = time.perf_counter()
            tasks = [measure(client, semaphore, "nutrition and health") for _ in range(count)]
            latencies = await asyncio.gather(*tasks)
            total_duration = time.perf_counter() - start_level
            
            valid_latencies = [l for l in latencies if l is not None]
            rps = len(valid_latencies) / total_duration
            
            for l in valid_latencies:
                all_data.append({"concurrency": level, "latency": l, "rps": rps})
    
    return pd.DataFrame(all_data)

async def main():
    df = await run_test()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Latency (ms)", "Throughput (RPS)"))
    
    colors = {1: "#636EFA", 2: "#EF553B", 4: "#00CC96", 8: "#AB63FA"}
    
    for level in CONFIG.keys():
        lvl_df = df[df["concurrency"] == level]
        fig.add_trace(go.Box(y=lvl_df["latency"], name=f"{level} Users", marker_color=colors[level]), row=1, col=1)
    
    rps_df = df.groupby("concurrency")["rps"].first().reset_index()
    fig.add_trace(go.Bar(x=[f"{int(l)} Users" for l in rps_df["concurrency"]], 
                         y=rps_df["rps"], 
                         text=rps_df["rps"].round(1),
                         marker_color="#330066"), row=1, col=2)

    fig.update_layout(title="Performance Stress Test", template="plotly_white", showlegend=False)
    fig.write_image("data/performance_final.png")
    print("Report saved: data/performance_final.png")

if __name__ == "__main__":
    asyncio.run(main())
