import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_mrr(rank):
    return 1.0 / rank if rank > 0 else 0.0

def create_advanced_dashboard():
    try:
        with open("data/benchmark_raw_data.json", "r") as f:
            raw_data = json.load(f)
    except:
        return

    df = pd.DataFrame(raw_data)
    df['mrr'] = df['target_rank'].apply(calculate_mrr)

    strat_names = {
        "log": "Лог. затухание (Log-Decay)",
        "linear": "Линейная (Linear)",
        "sigmoid": "Сигмоида (Sigmoid)"
    }
    
    colors = {
        "log": "#1f77b4",
        "linear": "#d62728",
        "sigmoid": "#2ca02c"
    }

    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=(
            "Рост MRR (Эффективность)", "Распределение позиций (Box)", "Гистограмма прироста позиций", "Доля попаданий в Топ-1",
            "MRR: Чистый vs Шумный", "Позиция ложного документа (Риск)", "Выживаемость целевого документа", "Коэффициент устойчивости",
            "Кривая обучения (0-20 кликов)", "Скорость сходимости (Ранг @ 3)", "Стабильность топа (Ранг @ 20)", "Финальный MRR"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    
    output_path = "data/final_advanced_dashboard_ru.html"
    fig.write_html(output_path)

if __name__ == "__main__":
    create_advanced_dashboard()
