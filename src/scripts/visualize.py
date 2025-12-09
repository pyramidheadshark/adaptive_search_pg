import plotly.graph_objects as go

def plot_results(data):
    fig = go.Figure()
    fig.add_trace(go.Bar(y=data))
    fig.write_html("results.html")
    print("Graph saved")

if __name__ == "__main__":
    plot_results([1, 2, 3])
