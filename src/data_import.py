import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

REQUIRED_COLUMNS = [
    '设备编号', '班次', '日期', '记录类型',
    '开始时间', '结束时间', '产量', '合格品数',
    '停机原因分类', '备注'
]

OPTIONAL_COLUMNS = ['标准节拍秒每件']

VALID_RECORD_TYPES = ['运行', '停机', '换模', '空转', '缺陷']
VALID_SHIFTS = ['早', '中', '晚']
VALID_DOWNTIME_REASONS = [
    '设备故障', '缺料', '质量', '计划维护', '换模调整', ''
]


def parse_datetime(date_str: str, time_str: str) -> datetime:
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError(f"无法解析时间: {date_str} {time_str}")


def parse_datetime_with_shift(date_str: str, start_time_str: str, 
                               end_time_str: str, shift: str = '') -> Tuple[datetime, datetime]:
    start_dt = parse_datetime(date_str, start_time_str)
    end_dt = parse_datetime(date_str, end_time_str)
    
    if shift == '晚':
        if start_dt.hour < 12:
            start_dt += timedelta(days=1)
        if end_dt.hour < 12:
            end_dt += timedelta(days=1)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
    
    return start_dt, end_dt


def is_valid_time_order(start_time_str: str, end_time_str: str, shift: str) -> bool:
    try:
        start_t = datetime.strptime(start_time_str, "%H:%M:%S")
        end_t = datetime.strptime(end_time_str, "%H:%M:%S")
    except ValueError:
        try:
            start_t = datetime.strptime(start_time_str, "%H:%M")
            end_t = datetime.strptime(end_time_str, "%H:%M")
        except ValueError:
            return False
    
    if shift == '晚':
        return True
    
    return end_t > start_t


def validate_csv(df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    errors = []
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"缺少必要列: {', '.join(missing_cols)}")
        return False, errors, df
    
    df = df.copy()
    df['_行号'] = df.index + 2
    
    for idx, row in df.iterrows():
        row_num = row['_行号']
        
        if pd.isna(row['设备编号']) or str(row['设备编号']).strip() == '':
            errors.append(f"第{row_num}行: 设备编号不能为空")
        
        if pd.isna(row['班次']) or row['班次'] not in VALID_SHIFTS:
            errors.append(f"第{row_num}行: 班次必须是 早/中/晚，当前值: {row['班次']}")
        
        if pd.isna(row['记录类型']) or row['记录类型'] not in VALID_RECORD_TYPES:
            errors.append(f"第{row_num}行: 记录类型必须是 运行/停机/换模/空转/缺陷，当前值: {row['记录类型']}")
            continue
        
        try:
            parse_datetime(str(row['日期']), str(row['开始时间']))
            parse_datetime(str(row['日期']), str(row['结束时间']))
        except ValueError as e:
            errors.append(f"第{row_num}行: {e}")
            continue
        
        if not is_valid_time_order(str(row['开始时间']), str(row['结束时间']), str(row['班次'])):
            errors.append(f"第{row_num}行: 结束时间必须大于开始时间（晚班允许跨天）")
        
        record_type = row['记录类型']
        
        if record_type == '运行':
            try:
                output = float(row['产量']) if pd.notna(row['产量']) else 0
                good_output = float(row['合格品数']) if pd.notna(row['合格品数']) else 0
            except (ValueError, TypeError):
                errors.append(f"第{row_num}行: 产量或合格品数格式错误")
                continue
            
            if output < 0:
                errors.append(f"第{row_num}行: 产量不能为负数")
            if good_output < 0:
                errors.append(f"第{row_num}行: 合格品数不能为负数")
            if good_output > output:
                errors.append(f"第{row_num}行: 合格品数不能超过产量")
        
        if record_type == '停机':
            reason = row['停机原因分类'] if pd.notna(row['停机原因分类']) else ''
            if reason and reason not in VALID_DOWNTIME_REASONS:
                errors.append(f"第{row_num}行: 停机原因分类无效: {reason}")
    
    has_time_errors = any('结束时间必须大于开始时间' in e or '无法解析时间' in e for e in errors)
    if not has_time_errors:
        overlap_errors = check_time_overlap(df)
        errors.extend(overlap_errors)
    
    return len(errors) == 0, errors, df


def check_time_overlap(df: pd.DataFrame) -> List[str]:
    errors = []
    df = df.copy()
    
    df['_开始时间'], df['_结束时间'] = zip(*df.apply(
        lambda r: parse_datetime_with_shift(
            str(r['日期']), str(r['开始时间']), str(r['结束时间']), str(r.get('班次', ''))
        ), axis=1
    ))
    
    for device_id, device_df in df.groupby('设备编号'):
        device_df = device_df.sort_values('_开始时间')
        intervals = []
        for idx, row in device_df.iterrows():
            intervals.append((row['_开始时间'], row['_结束时间'], row['_行号']))
        
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                start_i, end_i, row_i = intervals[i]
                start_j, end_j, row_j = intervals[j]
                if start_j < end_i and end_j > start_i:
                    errors.append(
                        f"设备{device_id}第{row_i}行与第{row_j}行时间段重叠: "
                        f"{start_i.strftime('%m-%d %H:%M')}-{end_i.strftime('%m-%d %H:%M')} "
                        f"与 {start_j.strftime('%m-%d %H:%M')}-{end_j.strftime('%m-%d %H:%M')}"
                    )
    
    return errors


def load_csv(filepath: str) -> Tuple[bool, List[str], pd.DataFrame]:
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='gbk')
    
    is_valid, errors, df = validate_csv(df)
    return is_valid, errors, df


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['开始时间戳'], df['结束时间戳'] = zip(*df.apply(
        lambda r: parse_datetime_with_shift(
            str(r['日期']), str(r['开始时间']), str(r['结束时间']), str(r.get('班次', ''))
        ), axis=1
    ))
    df['持续时间分钟'] = (df['结束时间戳'] - df['开始时间戳']).dt.total_seconds() / 60
    
    if '标准节拍秒每件' not in df.columns:
        df['标准节拍秒每件'] = np.nan
    
    df['产量'] = pd.to_numeric(df['产量'], errors='coerce').fillna(0)
    df['合格品数'] = pd.to_numeric(df['合格品数'], errors='coerce').fillna(0)
    df['停机原因分类'] = df['停机原因分类'].fillna('')
    
    return df


def get_device_list(df: pd.DataFrame) -> List[str]:
    return sorted(df['设备编号'].unique().tolist())


def get_date_range(df: pd.DataFrame) -> Tuple[str, str]:
    dates = df['日期'].unique()
    return str(min(dates)), str(max(dates))
