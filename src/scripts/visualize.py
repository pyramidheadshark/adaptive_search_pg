import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard():
    try:
        with open("data/benchmark_raw_data.json", "r") as f:
            data = json.load(f)
    except:
        print("No data found")
        return

    df = pd.DataFrame(data)
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(y=[1, 2, 3]))
    
    fig.write_html("dashboard.html")
    print("Dashboard created")

if __name__ == "__main__":
    create_dashboard()
