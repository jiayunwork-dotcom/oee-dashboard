import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple

from src.oee_engine import calculate_daily_oee, calculate_oee_for_device


def create_trend_chart(df: pd.DataFrame, device_ids: List[str],
                        start_date: str, end_date: str,
                        trend_type: str = '日',
                        configured_takts: Optional[Dict[str, float]] = None,
                        oee_target: float = 0.85,
                        show_moving_avg: bool = True) -> go.Figure:
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for idx, device_id in enumerate(device_ids):
        daily_data = calculate_daily_oee(df, device_id, 
                                          configured_takts.get(device_id) if configured_takts else None)
        
        if len(daily_data) == 0:
            continue
        
        daily_data = daily_data[
            (daily_data['日期'] >= start_date) & 
            (daily_data['日期'] <= end_date)
        ].copy()
        
        if trend_type == '周':
            daily_data['周'] = pd.to_datetime(daily_data['日期']).dt.isocalendar().week.astype(str)
            weekly_data = daily_data.groupby('周').agg(
                可用率=('可用率', 'mean'),
                性能率=('性能率', 'mean'),
                质量率=('质量率', 'mean'),
                OEE=('OEE', 'mean'),
            ).reset_index()
            weekly_data = weekly_data.sort_values('周')
            x_data = weekly_data['周']
            oee_data = weekly_data['OEE']
        elif trend_type == '月':
            daily_data['月'] = pd.to_datetime(daily_data['日期']).dt.to_period('M').astype(str)
            monthly_data = daily_data.groupby('月').agg(
                可用率=('可用率', 'mean'),
                性能率=('性能率', 'mean'),
                质量率=('质量率', 'mean'),
                OEE=('OEE', 'mean'),
            ).reset_index()
            monthly_data = monthly_data.sort_values('月')
            x_data = monthly_data['月']
            oee_data = monthly_data['OEE']
        else:
            x_data = daily_data['日期']
            oee_data = daily_data['OEE']
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=oee_data * 100,
                mode='lines+markers',
                name=f'{device_id} OEE',
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )
        
        if show_moving_avg and trend_type == '日' and len(daily_data) >= 7:
            ma7 = daily_data['OEE'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=daily_data['日期'],
                    y=ma7 * 100,
                    mode='lines',
                    name=f'{device_id} 7日移动平均',
                    line=dict(color=color, width=2, dash='dash'),
                    opacity=0.7,
                )
            )
    
    fig.add_hline(
        y=oee_target * 100,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"目标值: {oee_target*100:.0f}%",
        annotation_position="bottom right",
    )
    
    fig.update_layout(
        title=f'OEE{trend_type}趋势图',
        xaxis_title='时间',
        yaxis_title='OEE (%)',
        yaxis=dict(range=[0, 105]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def create_factor_trend_chart(df: pd.DataFrame, device_id: str,
                               start_date: str, end_date: str,
                               configured_takt: Optional[float] = None,
                               trend_type: str = '日') -> go.Figure:
    daily_data = calculate_daily_oee(df, device_id, configured_takt)
    
    if len(daily_data) == 0:
        fig = go.Figure()
        fig.update_layout(title='暂无数据')
        return fig
    
    daily_data = daily_data[
        (daily_data['日期'] >= start_date) & 
        (daily_data['日期'] <= end_date)
    ].copy()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['日期'],
            y=daily_data['可用率'] * 100,
            mode='lines+markers',
            name='可用率',
            line=dict(color='#3498db', width=2),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['日期'],
            y=daily_data['性能率'] * 100,
            mode='lines+markers',
            name='性能率',
            line=dict(color='#e67e22', width=2),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['日期'],
            y=daily_data['质量率'] * 100,
            mode='lines+markers',
            name='质量率',
            line=dict(color='#2ecc71', width=2),
        )
    )
    
    fig.update_layout(
        title=f'{device_id} OEE三因子趋势',
        xaxis_title='日期',
        yaxis_title='效率 (%)',
        yaxis=dict(range=[0, 105]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def create_device_comparison_chart(df: pd.DataFrame, start_date: str, end_date: str,
                                    configured_takts: Optional[Dict[str, float]] = None) -> go.Figure:
    devices = sorted(df['设备编号'].unique().tolist())
    
    oee_values = []
    availability_values = []
    performance_values = []
    quality_values = []
    
    for device in devices:
        takt = configured_takts.get(device) if configured_takts else None
        result = calculate_oee_for_device(df, device, start_date, end_date, takt)
        oee_values.append(result['OEE'] * 100)
        availability_values.append(result['可用率'] * 100)
        performance_values.append(result['性能率'] * 100)
        quality_values.append(result['质量率'] * 100)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=devices,
            y=oee_values,
            name='OEE',
            marker_color='#3498db',
            text=[f'{v:.1f}%' for v in oee_values],
            textposition='outside',
        )
    )
    
    fig.update_layout(
        title='各设备OEE排名对比',
        xaxis_title='设备编号',
        yaxis_title='OEE (%)',
        yaxis=dict(range=[0, 105]),
    )
    
    return fig


def create_shift_comparison_chart(df: pd.DataFrame, device_id: str,
                                   start_date: str, end_date: str,
                                   configured_takt: Optional[float] = None) -> go.Figure:
    shifts = ['早', '中', '晚']
    shift_names = ['早班', '中班', '晚班']
    
    oee_values = []
    availability_values = []
    performance_values = []
    quality_values = []
    
    for shift in shifts:
        result = calculate_oee_for_device(df, device_id, start_date, end_date, 
                                           configured_takt, shifts=[shift])
        oee_values.append(result['OEE'] * 100)
        availability_values.append(result['可用率'] * 100)
        performance_values.append(result['性能率'] * 100)
        quality_values.append(result['质量率'] * 100)
    
    fig = go.Figure()
    
    x = np.arange(len(shift_names))
    width = 0.2
    
    fig.add_trace(
        go.Bar(
            x=shift_names,
            y=availability_values,
            name='可用率',
            marker_color='#3498db',
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=shift_names,
            y=performance_values,
            name='性能率',
            marker_color='#e67e22',
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=shift_names,
            y=quality_values,
            name='质量率',
            marker_color='#2ecc71',
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=shift_names,
            y=oee_values,
            name='OEE',
            marker_color='#9b59b6',
        )
    )
    
    fig.update_layout(
        title=f'{device_id} 各班次OEE对比',
        xaxis_title='班次',
        yaxis_title='效率 (%)',
        yaxis=dict(range=[0, 105]),
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def create_benchmark_chart(oee: float, availability: float, 
                            performance: float, quality: float) -> go.Figure:
    categories = ['可用率', '性能率', '质量率', 'OEE']
    world_class = [90, 95, 99.9, 85]
    current = [availability * 100, performance * 100, quality * 100, oee * 100]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=world_class,
            name='世界级标杆',
            marker_color='#95a5a6',
            text=[f'{v:.1f}%' for v in world_class],
            textposition='outside',
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=current,
            name='当前水平',
            marker_color='#3498db',
            text=[f'{v:.1f}%' for v in current],
            textposition='outside',
        )
    )
    
    fig.update_layout(
        title='与世界级标杆对比',
        yaxis_title='效率 (%)',
        yaxis=dict(range=[0, 105]),
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def detect_anomalies(df: pd.DataFrame, device_id: str,
                      configured_takt: Optional[float] = None,
                      threshold_drop: float = 0.15,
                      oee_target: float = 0.85,
                      consecutive_days: int = 3) -> List[Dict]:
    daily_data = calculate_daily_oee(df, device_id, configured_takt)
    
    if len(daily_data) < 8:
        return []
    
    daily_data = daily_data.sort_values('日期').reset_index(drop=True)
    daily_data['OEE_ma7'] = daily_data['OEE'].rolling(window=7, min_periods=1).mean()
    
    anomalies = []
    
    for i in range(7, len(daily_data)):
        current_oee = daily_data.loc[i, 'OEE']
        prev_avg = daily_data.loc[i-1, 'OEE_ma7']
        drop = prev_avg - current_oee
        
        if drop >= threshold_drop:
            day_data = daily_data.iloc[i]
            factor_drop = max(
                ('可用率', daily_data.loc[i-1, '可用率'] - day_data['可用率']),
                ('性能率', daily_data.loc[i-1, '性能率'] - day_data['性能率']),
                ('质量率', daily_data.loc[i-1, '质量率'] - day_data['质量率']),
                key=lambda x: x[1]
            )
            
            anomalies.append({
                '类型': 'OEE骤降告警',
                '日期': day_data['日期'],
                '设备': device_id,
                '当前OEE': current_oee,
                '7日均值': prev_avg,
                '下降幅度': drop,
                '主要归因因子': factor_drop[0],
                '因子下降': factor_drop[1],
            })
    
    below_target_count = 0
    for i in range(len(daily_data)):
        if daily_data.loc[i, 'OEE'] < oee_target:
            below_target_count += 1
        else:
            below_target_count = 0
        
        if below_target_count >= consecutive_days:
            start_date = daily_data.loc[i - consecutive_days + 1, '日期']
            end_date = daily_data.loc[i, '日期']
            
            if not any(a.get('类型') == '持续低效预警' and a.get('结束日期') == end_date 
                       for a in anomalies):
                avg_oee = daily_data.loc[i - consecutive_days + 1:i, 'OEE'].mean()
                anomalies.append({
                    '类型': '持续低效预警',
                    '开始日期': start_date,
                    '结束日期': end_date,
                    '日期': end_date,
                    '设备': device_id,
                    '持续天数': consecutive_days,
                    '平均OEE': avg_oee,
                    '目标值': oee_target,
                })
    
    return anomalies


def get_all_anomalies(df: pd.DataFrame, 
                       configured_takts: Optional[Dict[str, float]] = None,
                       oee_target: float = 0.85) -> List[Dict]:
    all_anomalies = []
    devices = df['设备编号'].unique()
    
    for device in devices:
        takt = configured_takts.get(device) if configured_takts else None
        anomalies = detect_anomalies(df, device, takt, oee_target=oee_target)
        all_anomalies.extend(anomalies)
    
    return all_anomalies


def calculate_overall_daily_oee(df: pd.DataFrame, start_date: str, end_date: str,
                                 configured_takts: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    devices = df['设备编号'].unique()
    
    all_daily = []
    for device in devices:
        daily = calculate_daily_oee(df, device, 
                                     configured_takts.get(device) if configured_takts else None)
        if len(daily) > 0:
            daily['设备编号'] = device
            all_daily.append(daily)
    
    if not all_daily:
        return pd.DataFrame()
    
    combined = pd.concat(all_daily, ignore_index=True)
    combined = combined[
        (combined['日期'] >= start_date) & 
        (combined['日期'] <= end_date)
    ]
    
    combined['OEE_weighted'] = combined['OEE'] * combined['负荷时间']
    combined['可用性_weighted'] = combined['可用率'] * combined['负荷时间']
    combined['性能加权'] = combined['性能率'] * combined['负荷时间']
    combined['质量加权'] = combined['质量率'] * combined['负荷时间']
    
    daily_agg = combined.groupby('日期').agg({
        'OEE_weighted': 'sum',
        '可用性_weighted': 'sum',
        '性能加权': 'sum',
        '质量加权': 'sum',
        '负荷时间': 'sum',
    }).reset_index()
    
    daily_overall = pd.DataFrame()
    daily_overall['日期'] = daily_agg['日期']
    daily_overall['OEE'] = daily_agg['OEE_weighted'] / daily_agg['负荷时间'].where(daily_agg['负荷时间'] > 0, 1)
    daily_overall['可用率'] = daily_agg['可用性_weighted'] / daily_agg['负荷时间'].where(daily_agg['负荷时间'] > 0, 1)
    daily_overall['性能率'] = daily_agg['性能加权'] / daily_agg['负荷时间'].where(daily_agg['负荷时间'] > 0, 1)
    daily_overall['质量率'] = daily_agg['质量加权'] / daily_agg['负荷时间'].where(daily_agg['负荷时间'] > 0, 1)
    daily_overall['总负荷时间'] = daily_agg['负荷时间']
    
    return daily_overall
