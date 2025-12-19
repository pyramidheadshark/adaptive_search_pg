import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_mrr(rank):
    return 1.0 / rank if rank > 0 else 0.0

def create_advanced_dashboard():
    input_path = "data/benchmark_raw_data.json"
    try:
        with open(input_path, "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("Сначала запустите бенчмарк (make benchmark).")
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

    df_eff = df[(df['experiment'] == "Efficiency") & (df['clicks'] == 5)]
    df_eff_start = df[(df['experiment'] == "Efficiency") & (df['clicks'] == 0)]
    
    mrr_by_strat = df_eff.groupby('strategy')['mrr'].mean()
    for strat in strat_names.keys():
        if strat in mrr_by_strat:
            fig.add_trace(go.Bar(
                x=[strat_names[strat]], 
                y=[mrr_by_strat[strat]], 
                name=strat_names[strat],
                marker_color=colors[strat],
                legendgroup=strat
            ), row=1, col=1)

    for strat in df_eff['strategy'].unique():
        fig.add_trace(go.Box(
            y=df_eff[df_eff['strategy'] == strat]['target_rank'], 
            name=strat_names[strat], 
            marker_color=colors[strat], 
            legendgroup=strat,
            showlegend=False
        ), row=1, col=2)

    merged = pd.merge(df_eff_start, df_eff, on=['query_id', 'strategy'], suffixes=('_start', '_end'))
    merged['gain'] = merged['target_rank_start'] - merged['target_rank_end']
    for strat in merged['strategy'].unique():
        fig.add_trace(go.Histogram(
            x=merged[merged['strategy']==strat]['gain'], 
            name=strat_names[strat], 
            marker_color=colors[strat], 
            opacity=0.7, 
            legendgroup=strat,
            showlegend=False
        ), row=1, col=3)

    top1_rate = merged[merged['target_rank_end'] == 1].groupby('strategy').size() / merged.groupby('strategy').size()
    for s in colors.keys():
        if s not in top1_rate: top1_rate[s] = 0.0
    fig.add_trace(go.Bar(
        x=[strat_names[s] for s in colors.keys()],
        y=[top1_rate.get(s, 0) for s in colors.keys()],
        marker_color=[colors[s] for s in colors.keys()],
        showlegend=False
    ), row=1, col=4)

    df_noise = df[(df['experiment'] == "Noise") & (df['clicks'] == 5)]
    
    noise_qids = df_noise['query_id'].unique()
    df_eff_filtered = df_eff[df_eff['query_id'].isin(noise_qids)]
    
    clean_mrr = df_eff_filtered.groupby('strategy')['mrr'].mean()
    noisy_mrr = df_noise.groupby('strategy')['mrr'].mean()
    
    fig.add_trace(go.Bar(
        x=[strat_names[s] for s in noisy_mrr.index], 
        y=noisy_mrr.values, 
        marker_color=[colors[s] for s in noisy_mrr.index], 
        opacity=0.6, 
        showlegend=False
    ), row=2, col=1)

    for strat in df_noise['strategy'].unique():
        fig.add_trace(go.Box(
            y=df_noise[df_noise['strategy'] == strat]['distractor_rank'], 
            name=strat_names[strat], 
            marker_color=colors[strat], 
            legendgroup=strat,
            showlegend=False
        ), row=2, col=2)

    survival = df_noise[df_noise['target_rank'] <= 3].groupby('strategy').size() / df_noise.groupby('strategy').size()
    fig.add_trace(go.Bar(
        x=[strat_names[s] for s in survival.index], 
        y=survival.values, 
        marker_color=[colors[s] for s in survival.index], 
        showlegend=False
    ), row=2, col=3)

    resilience = {}
    for s in colors.keys():
        c = clean_mrr.get(s, 0.001)
        n = noisy_mrr.get(s, 0)
        resilience[s] = n / c if c > 0 else 0

    fig.add_trace(go.Bar(
        x=[strat_names[s] for s in colors.keys()], 
        y=[resilience[s] for s in colors.keys()], 
        marker_color=[colors[s] for s in colors.keys()], 
        showlegend=False
    ), row=2, col=4)


    df_sat = df[df['experiment'] == "Saturation"]
    
    learning = df_sat.groupby(['strategy', 'clicks'])['target_rank'].mean().reset_index()
    for strat in learning['strategy'].unique():
        subset = learning[learning['strategy'] == strat]
        fig.add_trace(go.Scatter(
            x=subset['clicks'], 
            y=subset['target_rank'], 
            mode='lines+markers', 
            name=strat_names[strat], 
            line=dict(color=colors[strat]),
            legendgroup=strat,
            showlegend=False
        ), row=3, col=1)

    at_click_3 = df_sat[df_sat['clicks'] == 3].groupby('strategy')['target_rank'].mean()
    fig.add_trace(go.Bar(
        x=[strat_names[s] for s in at_click_3.index], 
        y=at_click_3.values, 
        marker_color=[colors[s] for s in at_click_3.index], 
        showlegend=False
    ), row=3, col=2)

    at_click_20 = df_sat[df_sat['clicks'] == 20]
    for strat in at_click_20['strategy'].unique():
        fig.add_trace(go.Box(
            y=at_click_20[at_click_20['strategy'] == strat]['target_rank'], 
            name=strat_names[strat], 
            marker_color=colors[strat], 
            legendgroup=strat,
            showlegend=False
        ), row=3, col=3)

    final_mrr = at_click_20.groupby('strategy')['mrr'].mean()
    fig.add_trace(go.Bar(
        x=[strat_names[s] for s in final_mrr.index], 
        y=final_mrr.values, 
        marker_color=[colors[s] for s in final_mrr.index], 
        showlegend=False
    ), row=3, col=4)

    fig.update_layout(
        height=1200, 
        width=1600, 
        title_text="Аналитика Адаптивного Поиска: Сравнение Стратегий (v2)", 
        template="plotly_white", 
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="MRR (Выше лучше)", row=1, col=1)
    fig.update_yaxes(title_text="Позиция (Ниже лучше)", row=1, col=2)
    fig.update_yaxes(title_text="Кол-во запросов", row=1, col=3)
    fig.update_yaxes(title_text="Доля (0-1)", row=1, col=4)
    
    fig.update_yaxes(title_text="MRR", row=2, col=1)
    fig.update_yaxes(title_text="Позиция дистрактора", row=2, col=2)
    fig.update_yaxes(title_text="Выживаемость %", row=2, col=3)
    fig.update_yaxes(title_text="Коэфф. устойчивости", row=2, col=4)

    fig.update_xaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=True, title_text="Клики", row=3, col=1)

    output_path = "data/final_advanced_dashboard_ru.html"
    fig.write_html(output_path)
    print(f"Дашборд сгенерирован: {output_path}")

if __name__ == "__main__":
    create_advanced_dashboard()
