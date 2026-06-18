import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)


SHIFT_DEFS = {
    '早': {'start_hour': 6, 'end_hour': 14, 'break_hour': 9, 'break_minutes': 30},
    '中': {'start_hour': 14, 'end_hour': 22, 'break_hour': 17, 'break_minutes': 30},
    '晚': {'start_hour': 22, 'end_hour': 30, 'break_hour': 25, 'break_minutes': 30},
}


def generate_sample_data():
    devices = ['CNC-001', 'CNC-002', 'CNC-003', '冲压-001', '焊接-001']
    
    device_takt = {
        'CNC-001': 45,
        'CNC-002': 50,
        'CNC-003': 40,
        '冲压-001': 15,
        '焊接-001': 60,
    }
    
    device_oee_base = {
        'CNC-001': 0.75,
        'CNC-002': 0.68,
        'CNC-003': 0.82,
        '冲压-001': 0.62,
        '焊接-001': 0.58,
    }
    
    start_date = datetime(2025, 6, 2)
    end_date = datetime(2025, 6, 27)
    
    records = []
    
    anomaly_dates = [
        datetime(2025, 6, 6),
        datetime(2025, 6, 12),
        datetime(2025, 6, 18),
    ]
    
    low_performance_dates = [
        datetime(2025, 6, 9),
        datetime(2025, 6, 10),
        datetime(2025, 6, 11),
    ]
    
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        is_anomaly_day = current_date in anomaly_dates
        is_low_day = current_date in low_performance_dates
        
        for device in devices:
            for shift in ['早', '中', '晚']:
                base_oee = device_oee_base[device]
                
                if is_anomaly_day and device in ['CNC-001', '冲压-001']:
                    base_oee = base_oee * 0.6
                elif is_low_day:
                    base_oee = base_oee * 0.85
                
                daily_factor = 1 + random.uniform(-0.08, 0.05)
                base_oee = base_oee * daily_factor
                
                shift_events = generate_shift_events_clean(
                    current_date, shift, device,
                    device_takt[device],
                    base_oee
                )
                records.extend(shift_events)
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(records)
    df = df.sort_values(['设备编号', '日期', '开始时间']).reset_index(drop=True)
    
    return df


def generate_shift_events_clean(shift_date, shift_name, device, takt_time, base_oee):
    events = []
    shift_def = SHIFT_DEFS[shift_name]
    
    shift_start = datetime(shift_date.year, shift_date.month, shift_date.day,
                            shift_def['start_hour'] % 24, 0)
    shift_end = datetime(shift_date.year, shift_date.month, shift_date.day,
                         shift_def['end_hour'] % 24, 0)
    if shift_def['end_hour'] >= 24:
        shift_end += timedelta(days=1)
    
    break_start = datetime(shift_date.year, shift_date.month, shift_date.day,
                           shift_def['break_hour'] % 24, 0)
    if shift_def['break_hour'] >= 24:
        break_start += timedelta(days=1)
    break_end = break_start + timedelta(minutes=shift_def['break_minutes'])
    
    shift_duration_min = (shift_end - shift_start).total_seconds() / 60
    total_available = shift_duration_min - shift_def['break_minutes']
    
    availability = base_oee * 0.98 + random.uniform(-0.05, 0.03)
    performance = base_oee * 1.02 + random.uniform(-0.03, 0.03)
    quality = 0.98 + random.uniform(-0.02, 0.02)
    availability = min(max(availability, 0.5), 0.98)
    performance = min(max(performance, 0.6), 1.0)
    quality = min(max(quality, 0.9), 1.0)
    
    target_run = total_available * availability
    target_downtime = total_available - target_run
    
    segments = []
    
    before_break_start = shift_start
    before_break_end = break_start
    before_break_duration = (before_break_end - before_break_start).total_seconds() / 60
    
    after_break_start = break_end
    after_break_end = shift_end
    after_break_duration = (after_break_end - after_break_start).total_seconds() / 60
    
    if before_break_duration > 1:
        segments.extend(generate_segments_in_period(
            before_break_start, before_break_end,
            target_downtime * (before_break_duration / total_available),
            device, shift_name, shift_date, takt_time, performance, quality
        ))
    
    events.append({
        '设备编号': device,
        '班次': shift_name,
        '日期': shift_date.strftime('%Y-%m-%d'),
        '记录类型': '停机',
        '开始时间': fmt_time(break_start),
        '结束时间': fmt_time(break_end),
        '产量': '',
        '合格品数': '',
        '停机原因分类': '计划维护',
        '备注': '计划休息',
    })
    
    if after_break_duration > 1:
        segments.extend(generate_segments_in_period(
            after_break_start, after_break_end,
            target_downtime * (after_break_duration / total_available),
            device, shift_name, shift_date, takt_time, performance, quality
        ))
    
    events.extend(segments)
    
    events.sort(key=lambda x: x['开始时间'])
    
    return events


def generate_segments_in_period(period_start, period_end, target_downtime,
                                 device, shift_name, shift_date, takt_time, 
                                 performance, quality):
    segments = []
    period_duration = (period_end - period_start).total_seconds() / 60
    
    if period_duration < 5:
        output = period_duration * 60 / takt_time * performance
        good = output * quality
        segments.append({
            '设备编号': device,
            '班次': shift_name,
            '日期': shift_date.strftime('%Y-%m-%d'),
            '记录类型': '运行',
            '开始时间': fmt_time(period_start),
            '结束时间': fmt_time(period_end),
            '产量': round(output, 1),
            '合格品数': round(good, 1),
            '停机原因分类': '',
            '备注': '',
        })
        return segments
    
    num_downtimes = random.randint(1, 3)
    actual_downtime = 0
    downtime_events = []
    
    for i in range(num_downtimes):
        remaining_dt = target_downtime - actual_downtime
        if remaining_dt <= 2:
            break
        
        reasons = ['设备故障', '缺料', '质量', '换模调整', '计划维护', '']
        weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
        reason = random.choices(reasons, weights=weights)[0]
        
        if reason == '':
            dur = random.uniform(1, 5)
        elif reason == '计划维护':
            dur = random.uniform(10, min(remaining_dt * 0.3, 30))
        else:
            dur = random.uniform(3, min(remaining_dt * 0.4, 30))
        
        dur = min(dur, remaining_dt)
        actual_downtime += dur
        downtime_events.append({'reason': reason, 'duration': dur})
    
    total_run = period_duration - sum(d['duration'] for d in downtime_events)
    
    if total_run <= 0:
        output = period_duration * 60 / takt_time * performance
        good = output * quality
        segments.append({
            '设备编号': device,
            '班次': shift_name,
            '日期': shift_date.strftime('%Y-%m-%d'),
            '记录类型': '运行',
            '开始时间': fmt_time(period_start),
            '结束时间': fmt_time(period_end),
            '产量': round(output, 1),
            '合格品数': round(good, 1),
            '停机原因分类': '',
            '备注': '',
        })
        return segments
    
    num_run_segments = len(downtime_events) + 1
    run_segment_duration = total_run / num_run_segments
    
    current_time = period_start
    
    for i in range(num_run_segments):
        run_end = current_time + timedelta(minutes=run_segment_duration)
        
        if run_end > period_end:
            run_end = period_end
        
        actual_run_min = (run_end - current_time).total_seconds() / 60
        
        if actual_run_min > 0.5:
            output = actual_run_min * 60 / takt_time * performance
            good = output * quality
            segments.append({
                '设备编号': device,
                '班次': shift_name,
                '日期': shift_date.strftime('%Y-%m-%d'),
                '记录类型': '运行',
                '开始时间': fmt_time(current_time),
                '结束时间': fmt_time(run_end),
                '产量': round(output, 1),
                '合格品数': round(good, 1),
                '停机原因分类': '',
                '备注': '',
            })
        
        current_time = run_end
        
        if i < len(downtime_events):
            dt_info = downtime_events[i]
            dt_end = current_time + timedelta(minutes=dt_info['duration'])
            
            if dt_end > period_end:
                dt_end = period_end
            
            actual_dt_min = (dt_end - current_time).total_seconds() / 60
            
            if actual_dt_min >= 0.5:
                record_type = '换模' if dt_info['reason'] == '换模调整' else '停机'
                segments.append({
                    '设备编号': device,
                    '班次': shift_name,
                    '日期': shift_date.strftime('%Y-%m-%d'),
                    '记录类型': record_type,
                    '开始时间': fmt_time(current_time),
                    '结束时间': fmt_time(dt_end),
                    '产量': '',
                    '合格品数': '',
                    '停机原因分类': dt_info['reason'],
                    '备注': '',
                })
            
            current_time = dt_end
    
    return segments


def fmt_time(dt):
    return f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sample_data.csv')
    
    print("正在生成示例数据...")
    df = generate_sample_data()
    
    print(f"共生成 {len(df)} 条记录")
    print(f"设备数量: {df['设备编号'].nunique()}")
    print(f"日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存到: {output_path}")
    
    print("\n前10条记录:")
    print(df.head(10).to_string())


if __name__ == '__main__':
    main()
