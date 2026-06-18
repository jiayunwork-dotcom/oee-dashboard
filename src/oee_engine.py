import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

SHIFT_HOURS = {
    '早': {'start': (6, 0), 'end': (14, 0), 'break_minutes': 30},
    '中': {'start': (14, 0), 'end': (22, 0), 'break_minutes': 30},
    '晚': {'start': (22, 0), 'end': (6, 0), 'break_minutes': 30},
}

STANDARD_TAKT = {
    '默认': 60,
}


def set_standard_takt(device_model: str, takt_seconds: float):
    STANDARD_TAKT[device_model] = takt_seconds


def get_shift_duration_minutes(shift: str) -> float:
    info = SHIFT_HOURS.get(shift, SHIFT_HOURS['早'])
    start_h, start_m = info['start']
    end_h, end_m = info['end']
    
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    
    if end_minutes <= start_minutes:
        end_minutes += 24 * 60
    
    return end_minutes - start_minutes


def get_planned_break_minutes(shift: str) -> float:
    return SHIFT_HOURS.get(shift, SHIFT_HOURS['早'])['break_minutes']


def calculate_takt_time(df: pd.DataFrame, device_id: str, 
                         configured_takt: Optional[float] = None) -> Tuple[float, bool]:
    if configured_takt is not None and configured_takt > 0:
        return configured_takt, True
    
    device_data = df[
        (df['设备编号'] == device_id) & 
        (df['记录类型'] == '运行')
    ].copy()
    
    if len(device_data) == 0:
        return STANDARD_TAKT.get('默认', 60), False
    
    csv_takt = device_data['标准节拍秒每件'].dropna()
    if len(csv_takt) > 0:
        return csv_takt.iloc[0], True
    
    device_data['实际节拍'] = (device_data['持续时间分钟'] * 60) / device_data['产量'].replace(0, np.nan)
    valid_takt = device_data['实际节拍'].dropna()
    
    if len(valid_takt) > 0:
        best_takt = valid_takt.min()
        return best_takt, False
    
    return STANDARD_TAKT.get('默认', 60), False


def split_record_by_shift(record: pd.Series) -> List[pd.Series]:
    start = record['开始时间戳']
    end = record['结束时间戳']
    
    shift_boundaries = []
    current = start.replace(hour=6, minute=0, second=0, microsecond=0)
    for _ in range(3):
        shift_boundaries.append(current)
        current += timedelta(hours=8)
    
    pieces = []
    seg_start = start
    
    for boundary in shift_boundaries:
        if seg_start < boundary < end:
            piece = record.copy()
            piece['开始时间戳'] = seg_start
            piece['结束时间戳'] = boundary
            piece['持续时间分钟'] = (boundary - seg_start).total_seconds() / 60
            
            if record['记录类型'] == '运行':
                ratio = piece['持续时间分钟'] / record['持续时间分钟']
                piece['产量'] = record['产量'] * ratio
                piece['合格品数'] = record['合格品数'] * ratio
            
            pieces.append(piece)
            seg_start = boundary
    
    if seg_start < end:
        piece = record.copy()
        piece['开始时间戳'] = seg_start
        piece['结束时间戳'] = end
        piece['持续时间分钟'] = (end - seg_start).total_seconds() / 60
        
        if record['记录类型'] == '运行':
            ratio = piece['持续时间分钟'] / record['持续时间分钟']
            piece['产量'] = record['产量'] * ratio
            piece['合格品数'] = record['合格品数'] * ratio
        
        pieces.append(piece)
    
    return pieces if pieces else [record]


def classify_six_losses(record: pd.Series, takt_seconds: float, 
                         is_startup: bool = False) -> Dict[str, float]:
    losses = {
        '设备故障损失': 0,
        '换模调整损失': 0,
        '空转短停损失': 0,
        '速度降低损失': 0,
        '工艺缺陷损失': 0,
        '启动损失': 0,
    }
    
    record_type = record['记录类型']
    duration = record['持续时间分钟']
    
    if record_type == '停机':
        reason = record.get('停机原因分类', '')
        if reason == '设备故障':
            losses['设备故障损失'] = duration
        elif reason == '换模调整':
            losses['换模调整损失'] = duration
        elif reason in ['计划维护', '']:
            if duration < 5 and reason == '':
                losses['空转短停损失'] = duration
            elif reason == '':
                losses['空转短停损失'] = duration
        else:
            losses['设备故障损失'] = duration
    
    elif record_type == '换模':
        losses['换模调整损失'] = duration
    
    elif record_type == '空转':
        losses['空转短停损失'] = duration
    
    elif record_type == '运行':
        output = record['产量']
        if output > 0:
            actual_cycle = (duration * 60) / output
            if actual_cycle > takt_seconds:
                speed_loss_minutes = (actual_cycle - takt_seconds) * output / 60
                losses['速度降低损失'] = min(speed_loss_minutes, duration)
        
        defects = output - record['合格品数']
        if defects > 0:
            losses['工艺缺陷损失'] = (defects * takt_seconds) / 60
    
    if is_startup and record_type == '运行':
        startup_loss = duration * 0.3
        losses['启动损失'] += startup_loss
    
    return losses


def calculate_oee_for_device(df: pd.DataFrame, device_id: str, 
                              start_date: str, end_date: str,
                              configured_takt: Optional[float] = None,
                              shifts: Optional[List[str]] = None) -> Dict:
    device_df = df[df['设备编号'] == device_id].copy()
    device_df = device_df[
        (device_df['日期'] >= start_date) & 
        (device_df['日期'] <= end_date)
    ]
    
    if shifts:
        device_df = device_df[device_df['班次'].isin(shifts)]
    
    if len(device_df) == 0:
        return {
            '设备编号': device_id,
            '计划生产时间': 0,
            '负荷时间': 0,
            '运转时间': 0,
            '净运转时间': 0,
            '价值运转时间': 0,
            '可用率': 0,
            '性能率': 0,
            '质量率': 0,
            'OEE': 0,
            '总产量': 0,
            '合格品数': 0,
            '节拍时间': 0,
            '节拍是标准值': False,
        }
    
    takt_seconds, is_standard = calculate_takt_time(df, device_id, configured_takt)
    
    shift_dates = device_df[['日期', '班次']].drop_duplicates()
    
    planned_production_time = 0
    planned_maintenance_time = 0
    
    for _, row in shift_dates.iterrows():
        shift_duration = get_shift_duration_minutes(row['班次'])
        break_time = get_planned_break_minutes(row['班次'])
        planned_production_time += shift_duration - break_time
    
    maintenance_df = device_df[device_df['停机原因分类'] == '计划维护']
    planned_maintenance_time = maintenance_df['持续时间分钟'].sum()
    
    load_time = planned_production_time - planned_maintenance_time
    
    run_df = device_df[device_df['记录类型'] == '运行']
    run_time = run_df['持续时间分钟'].sum()
    
    total_output = run_df['产量'].sum()
    total_good = run_df['合格品数'].sum()
    
    net_run_time = (takt_seconds * total_output) / 60 if total_output > 0 else 0
    value_run_time = (takt_seconds * total_good) / 60 if total_good > 0 else 0
    
    availability = run_time / load_time if load_time > 0 else 0
    performance = net_run_time / run_time if run_time > 0 else 0
    quality = total_good / total_output if total_output > 0 else 0
    
    oee = availability * performance * quality
    
    return {
        '设备编号': device_id,
        '计划生产时间': planned_production_time,
        '计划维护时间': planned_maintenance_time,
        '负荷时间': load_time,
        '运转时间': run_time,
        '净运转时间': net_run_time,
        '价值运转时间': value_run_time,
        '可用率': availability,
        '性能率': performance,
        '质量率': quality,
        'OEE': oee,
        '总产量': total_output,
        '合格品数': total_good,
        '节拍时间': takt_seconds,
        '节拍是标准值': is_standard,
    }


def calculate_oee_overall(df: pd.DataFrame, start_date: str, end_date: str,
                           configured_takts: Optional[Dict[str, float]] = None,
                           shifts: Optional[List[str]] = None) -> Dict:
    devices = df['设备编号'].unique()
    device_results = {}
    
    for device in devices:
        takt = configured_takts.get(device) if configured_takts else None
        result = calculate_oee_for_device(df, device, start_date, end_date, takt, shifts)
        device_results[device] = result
    
    total_load_time = sum(r['负荷时间'] for r in device_results.values())
    
    weighted_availability = sum(r['可用率'] * r['负荷时间'] for r in device_results.values()) / total_load_time if total_load_time > 0 else 0
    weighted_performance = sum(r['性能率'] * r['负荷时间'] for r in device_results.values()) / total_load_time if total_load_time > 0 else 0
    weighted_quality = sum(r['质量率'] * r['负荷时间'] for r in device_results.values()) / total_load_time if total_load_time > 0 else 0
    
    weighted_oee = weighted_availability * weighted_performance * weighted_quality
    
    return {
        '设备汇总': {
            '可用率': weighted_availability,
            '性能率': weighted_performance,
            '质量率': weighted_quality,
            'OEE': weighted_oee,
            '总负荷时间': total_load_time,
        },
        '各设备': device_results,
    }


def calculate_daily_oee(df: pd.DataFrame, device_id: str, 
                         configured_takt: Optional[float] = None) -> pd.DataFrame:
    device_df = df[df['设备编号'] == device_id].copy()
    
    if len(device_df) == 0:
        return pd.DataFrame()
    
    dates = sorted(device_df['日期'].unique())
    daily_results = []
    
    for date in dates:
        result = calculate_oee_for_device(df, device_id, date, date, configured_takt)
        daily_results.append({
            '日期': date,
            '可用率': result['可用率'],
            '性能率': result['性能率'],
            '质量率': result['质量率'],
            'OEE': result['OEE'],
            '负荷时间': result['负荷时间'],
            '运转时间': result['运转时间'],
            '总产量': result['总产量'],
            '合格品数': result['合格品数'],
        })
    
    return pd.DataFrame(daily_results)


def calculate_six_losses_summary(df: pd.DataFrame, device_id: str,
                                  start_date: str, end_date: str,
                                  configured_takt: Optional[float] = None,
                                  shifts: Optional[List[str]] = None) -> Dict[str, float]:
    device_df = df[df['设备编号'] == device_id].copy()
    device_df = device_df[
        (device_df['日期'] >= start_date) & 
        (device_df['日期'] <= end_date)
    ]
    
    if shifts:
        device_df = device_df[device_df['班次'].isin(shifts)]
    
    takt_seconds, _ = calculate_takt_time(df, device_id, configured_takt)
    
    total_losses = {
        '设备故障损失': 0,
        '换模调整损失': 0,
        '空转短停损失': 0,
        '速度降低损失': 0,
        '工艺缺陷损失': 0,
        '启动损失': 0,
    }
    
    for _, record in device_df.iterrows():
        losses = classify_six_losses(record, takt_seconds)
        for key, value in losses.items():
            total_losses[key] += value
    
    return total_losses


def downtime_by_category(df: pd.DataFrame, start_date: str, end_date: str,
                          devices: Optional[List[str]] = None) -> pd.DataFrame:
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
        return pd.DataFrame(columns=['停机原因', '持续时间分钟', '次数'])
    
    downtime['停机原因'] = downtime.apply(
        lambda r: r['停机原因分类'] if r['记录类型'] == '停机' else '换模调整',
        axis=1
    )
    
    result = downtime.groupby('停机原因').agg(
        持续时间分钟=('持续时间分钟', 'sum'),
        次数=('持续时间分钟', 'count')
    ).reset_index()
    
    result = result.sort_values('持续时间分钟', ascending=False)
    return result
