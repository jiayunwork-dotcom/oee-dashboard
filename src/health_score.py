import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from src.oee_engine import calculate_daily_oee, calculate_oee_for_device


HEALTH_LEVELS = [
    {
        'name': '优秀',
        'min': 90,
        'max': 100,
        'color': '#198754',
        'bg_color': '#d1e7dd',
    },
    {
        'name': '良好',
        'min': 70,
        'max': 89,
        'color': '#0d6efd',
        'bg_color': '#cfe2ff',
    },
    {
        'name': '关注',
        'min': 50,
        'max': 69,
        'color': '#fd7e14',
        'bg_color': '#ffe5d0',
    },
    {
        'name': '警告',
        'min': 0,
        'max': 49,
        'color': '#dc3545',
        'bg_color': '#f8d7da',
    },
]


def get_health_level(score: int) -> Dict:
    for level in HEALTH_LEVELS:
        if level['min'] <= score <= level['max']:
            return level
    return HEALTH_LEVELS[-1]


def calculate_health_score_for_device(
    df: pd.DataFrame,
    device_id: str,
    end_date: str,
    configured_takt: Optional[float] = None,
    window_days: int = 7
) -> Dict:
    daily_data = calculate_daily_oee(df, device_id, configured_takt)
    
    if len(daily_data) == 0:
        return {
            '设备编号': device_id,
            '健康分': 0,
            '可用率': 0,
            '性能率': 0,
            '质量率': 0,
            '基础分': 0,
            '稳定性惩罚': 0,
            '数据天数': 0,
            '数据充足': False,
        }
    
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=window_days - 1)
    start_date_str = start_dt.strftime('%Y-%m-%d')
    
    window_data = daily_data[
        (daily_data['日期'] >= start_date_str) & 
        (daily_data['日期'] <= end_date)
    ].copy()
    
    if len(window_data) == 0:
        return {
            '设备编号': device_id,
            '健康分': 0,
            '可用率': 0,
            '性能率': 0,
            '质量率': 0,
            '基础分': 0,
            '稳定性惩罚': 0,
            '数据天数': 0,
            '数据充足': False,
        }
    
    avg_availability = window_data['可用率'].mean()
    avg_performance = window_data['性能率'].mean()
    avg_quality = window_data['质量率'].mean()
    
    base_score = avg_availability * 40 + avg_performance * 35 + avg_quality * 25
    
    oee_values = window_data['OEE'].values
    if len(oee_values) >= 2:
        oee_std = np.std(oee_values, ddof=1)
    else:
        oee_std = 0
    
    stability_penalty = oee_std * 50
    
    final_score = max(0, base_score - stability_penalty)
    final_score = round(final_score)
    
    data_sufficient = len(window_data) >= window_days
    
    return {
        '设备编号': device_id,
        '健康分': final_score,
        '可用率': avg_availability,
        '性能率': avg_performance,
        '质量率': avg_quality,
        '基础分': base_score,
        '稳定性惩罚': stability_penalty,
        '数据天数': len(window_data),
        '数据充足': data_sufficient,
    }


def calculate_all_health_scores(
    df: pd.DataFrame,
    end_date: str,
    configured_takts: Optional[Dict[str, float]] = None,
    window_days: int = 7
) -> Dict[str, Dict]:
    devices = sorted(df['设备编号'].unique().tolist())
    
    results = {}
    for device in devices:
        takt = configured_takts.get(device) if configured_takts else None
        result = calculate_health_score_for_device(df, device, end_date, takt, window_days)
        results[device] = result
    
    return results


def calculate_health_score_trend(
    df: pd.DataFrame,
    device_id: str,
    start_date: str,
    end_date: str,
    configured_takt: Optional[float] = None,
    window_days: int = 7
) -> pd.DataFrame:
    daily_data = calculate_daily_oee(df, device_id, configured_takt)
    
    if len(daily_data) == 0:
        return pd.DataFrame()
    
    daily_data = daily_data.sort_values('日期').reset_index(drop=True)
    
    trend_data = []
    
    for i in range(len(daily_data)):
        current_date = daily_data.loc[i, '日期']
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        window_start_dt = current_dt - timedelta(days=window_days - 1)
        window_start_str = window_start_dt.strftime('%Y-%m-%d')
        
        window_data = daily_data[
            (daily_data['日期'] >= window_start_str) & 
            (daily_data['日期'] <= current_date)
        ]
        
        if len(window_data) == 0:
            continue
        
        avg_availability = window_data['可用率'].mean()
        avg_performance = window_data['性能率'].mean()
        avg_quality = window_data['质量率'].mean()
        
        base_score = avg_availability * 40 + avg_performance * 35 + avg_quality * 25
        
        oee_values = window_data['OEE'].values
        if len(oee_values) >= 2:
            oee_std = np.std(oee_values, ddof=1)
        else:
            oee_std = 0
        
        stability_penalty = oee_std * 50
        final_score = max(0, base_score - stability_penalty)
        final_score = round(final_score)
        
        trend_data.append({
            '日期': current_date,
            '健康分': final_score,
            '可用率': avg_availability,
            '性能率': avg_performance,
            '质量率': avg_quality,
            '基础分': base_score,
            '稳定性惩罚': stability_penalty,
            '数据天数': len(window_data),
        })
    
    result_df = pd.DataFrame(trend_data)
    
    if len(result_df) > 0:
        result_df = result_df[
            (result_df['日期'] >= start_date) & 
            (result_df['日期'] <= end_date)
        ]
    
    return result_df


def detect_health_score_drop_anomalies(
    df: pd.DataFrame,
    current_scores: Dict[str, Dict],
    previous_scores: Dict[str, Dict],
    drop_threshold: int = 15
) -> List[Dict]:
    anomalies = []
    
    for device_id, current in current_scores.items():
        previous = previous_scores.get(device_id)
        if not previous:
            continue
        
        current_score = current['健康分']
        previous_score = previous['健康分']
        drop = previous_score - current_score
        
        if drop >= drop_threshold:
            avail_drop = previous['可用率'] - current['可用率']
            perf_drop = previous['性能率'] - current['性能率']
            qual_drop = previous['质量率'] - current['质量率']
            
            factor_drops = [
                ('可用率', avail_drop, 40),
                ('性能率', perf_drop, 35),
                ('质量率', qual_drop, 25),
            ]
            
            max_factor = max(factor_drops, key=lambda x: x[1] * x[2])
            
            anomalies.append({
                '类型': '健康分骤降',
                '设备': device_id,
                '上次评分': previous_score,
                '本次评分': current_score,
                '下降幅度': drop,
                '主要归因因子': max_factor[0],
                '因子下降': max_factor[1],
                '因子权重': max_factor[2],
            })
    
    return anomalies
