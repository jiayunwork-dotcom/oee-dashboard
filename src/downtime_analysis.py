import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional

from src.oee_engine import downtime_by_category

COLORS = {
    '运行': '#2ecc71',
    '停机': '#e74c3c',
    '故障停机': '#e74c3c',
    '换模': '#f39c12',
    '空转': '#f1c40f',
    '计划维护': '#95a5a6',
    '缺陷': '#e67e22',
}

DOWNTIME_COLORS = {
    '设备故障': '#e74c3c',
    '缺料': '#e67e22',
    '质量': '#9b59b6',
    '计划维护': '#95a5a6',
    '换模调整': '#f39c12',
    '空转短停': '#f1c40f',
}


def create_pareto_chart(df: pd.DataFrame, start_date: str, end_date: str,
                         devices: Optional[List[str]] = None) -> go.Figure:
    downtime_df = downtime_by_category(df, start_date, end_date, devices)
    
    if len(downtime_df) == 0:
        fig = go.Figure()
        fig.update_layout(title='暂无停机数据')
        return fig
    
    downtime_df = downtime_df.sort_values('持续时间分钟', ascending=False)
    downtime_df['累积百分比'] = downtime_df['持续时间分钟'].cumsum() / downtime_df['持续时间分钟'].sum() * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    colors = [DOWNTIME_COLORS.get(cat, '#3498db') for cat in downtime_df['停机原因']]
    
    fig.add_trace(
        go.Bar(
            x=downtime_df['停机原因'],
            y=downtime_df['持续时间分钟'],
            name='停机时间(分钟)',
            marker_color=colors,
            text=downtime_df['持续时间分钟'].round(1),
            textposition='outside',
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=downtime_df['停机原因'],
            y=downtime_df['累积百分比'],
            name='累积百分比',
            mode='lines+markers',
            line=dict(color='#2c3e50', width=2),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=80,
        x1=len(downtime_df) - 0.5,
        y1=80,
        line=dict(color="#e74c3c", width=2, dash="dash"),
        secondary_y=True,
    )
    
    fig.add_annotation(
        x=len(downtime_df) - 1,
        y=80,
        text="80% 分界线",
        showarrow=False,
        yshift=10,
        secondary_y=True,
        font=dict(color="#e74c3c"),
    )
    
    fig.update_layout(
        title='停机原因Pareto分析',
        barmode='group',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig.update_yaxes(title_text="停机时间(分钟)", secondary_y=False)
    fig.update_yaxes(title_text="累积百分比(%)", secondary_y=True, range=[0, 105])
    
    return fig


def create_gantt_chart(df: pd.DataFrame, start_date: str, end_date: str,
                        devices: Optional[List[str]] = None) -> go.Figure:
    filtered = df[
        (df['日期'] >= start_date) & 
        (df['日期'] <= end_date)
    ].copy()
    
    if devices:
        filtered = filtered[filtered['设备编号'].isin(devices)]
    
    if len(filtered) == 0:
        fig = go.Figure()
        fig.update_layout(title='暂无数据')
        return fig
    
    device_list = sorted(filtered['设备编号'].unique())
    
    fig = go.Figure()
    
    for device_idx, device in enumerate(device_list):
        device_data = filtered[filtered['设备编号'] == device].sort_values('开始时间戳')
        
        for _, row in device_data.iterrows():
            record_type = row['记录类型']
            color = COLORS.get(record_type, '#3498db')
            
            if record_type == '停机':
                reason = row['停机原因分类']
                if reason:
                    color = DOWNTIME_COLORS.get(reason, '#e74c3c')
                label = f"停机: {reason if reason else '未分类'}"
            else:
                type_names = {
                    '运行': '运行',
                    '换模': '换模',
                    '空转': '空转',
                    '缺陷': '缺陷',
                }
                label = type_names.get(record_type, record_type)
            
            hover_text = (
                f"设备: {device}<br>"
                f"类型: {label}<br>"
                f"开始: {row['开始时间戳'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"结束: {row['结束时间戳'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"持续: {row['持续时间分钟']:.1f}分钟"
            )
            
            if record_type == '运行':
                hover_text += (
                    f"<br>产量: {row['产量']:.0f}<br>"
                    f"合格品: {row['合格品数']:.0f}"
                )
            
            fig.add_trace(
                go.Scatter(
                    x=[row['开始时间戳'], row['结束时间戳']],
                    y=[device_idx, device_idx],
                    mode='lines',
                    line=dict(color=color, width=20),
                    name=label,
                    showlegend=False,
                    hovertext=hover_text,
                    hoverinfo='text',
                )
            )
    
    fig.update_yaxes(
        tickvals=list(range(len(device_list))),
        ticktext=device_list,
        autorange='reversed',
    )
    
    fig.update_layout(
        title='设备运行时间轴甘特图',
        xaxis_title='时间',
        yaxis_title='设备',
        height=max(300, len(device_list) * 80),
        hovermode='closest',
    )
    
    legend_items = [
        ('运行', COLORS['运行']),
        ('设备故障', DOWNTIME_COLORS['设备故障']),
        ('换模调整', DOWNTIME_COLORS['换模调整']),
        ('空转短停', DOWNTIME_COLORS['空转短停']),
        ('计划维护', DOWNTIME_COLORS['计划维护']),
        ('缺料', DOWNTIME_COLORS['缺料']),
        ('质量', DOWNTIME_COLORS['质量']),
    ]
    
    for name, color in legend_items:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=color, width=10),
                name=name,
                showlegend=True,
            )
        )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def create_downtime_pie(df: pd.DataFrame, start_date: str, end_date: str,
                         dimension: str = '设备',
                         devices: Optional[List[str]] = None) -> go.Figure:
    filtered = df[
        (df['日期'] >= start_date) & 
        (df['日期'] <= end_date)
    ].copy()
    
    if devices:
        filtered = filtered[filtered['设备编号'].isin(devices)]
    
    downtime = filtered[
        (filtered['记录类型'] == '停机') | 
        (filtered['记录类型'] == '换模')
    ].copy()
    
    if len(downtime) == 0:
        fig = go.Figure()
        fig.update_layout(title='暂无停机数据')
        return fig
    
    downtime['停机原因'] = downtime.apply(
        lambda r: r['停机原因分类'] if r['记录类型'] == '停机' else '换模调整',
        axis=1
    )
    
    if dimension == '设备':
        fig_data = downtime.groupby('设备编号')['持续时间分钟'].sum().reset_index()
        fig = px.pie(fig_data, names='设备编号', values='持续时间分钟',
                     title='各设备停机时间占比',
                     color_discrete_sequence=px.colors.qualitative.Set3)
    elif dimension == '班次':
        fig_data = downtime.groupby('班次')['持续时间分钟'].sum().reset_index()
        fig = px.pie(fig_data, names='班次', values='持续时间分钟',
                     title='各班次停机时间占比',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    elif dimension == '日期':
        fig_data = downtime.groupby('日期')['持续时间分钟'].sum().reset_index()
        fig = px.pie(fig_data, names='日期', values='持续时间分钟',
                     title='各日期停机时间占比',
                     color_discrete_sequence=px.colors.qualitative.Bold)
    else:
        fig_data = downtime.groupby('停机原因')['持续时间分钟'].sum().reset_index()
        fig = px.pie(fig_data, names='停机原因', values='持续时间分钟',
                     title='停机原因分类占比')
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    
    return fig


def create_root_cause_drilldown(df: pd.DataFrame, start_date: str, end_date: str,
                                 devices: Optional[List[str]] = None) -> Dict:
    from src.oee_engine import calculate_oee_overall
    
    overall = calculate_oee_overall(df, start_date, end_date)
    
    devices_data = overall['各设备']
    device_list = sorted(devices_data.keys())
    
    factor_analysis = []
    for device in device_list:
        d = devices_data[device]
        factor_analysis.append({
            '设备': device,
            'OEE': d['OEE'],
            '可用率': d['可用率'],
            '性能率': d['性能率'],
            '质量率': d['质量率'],
            '可用率贡献': d['可用率'],
            '性能率贡献': d['性能率'],
            '质量率贡献': d['质量率'],
        })
    
    pareto_data = downtime_by_category(df, start_date, end_date, devices)
    
    return {
        'overall_oee': overall['设备汇总']['OEE'],
        'overall_availability': overall['设备汇总']['可用率'],
        'overall_performance': overall['设备汇总']['性能率'],
        'overall_quality': overall['设备汇总']['质量率'],
        'device_details': factor_analysis,
        'pareto_data': pareto_data.to_dict('records') if len(pareto_data) > 0 else [],
    }
