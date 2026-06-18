import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.special import gamma


SHIFT_HOURS = {
    '早': {'start': (6, 0), 'end': (14, 0)},
    '中': {'start': (14, 0), 'end': (22, 0)},
    '晚': {'start': (22, 0), 'end': (6, 0)},
}

MORNING_MAINTENANCE_WINDOW = (6, 0, 6, 30)
EVENING_MAINTENANCE_WINDOW = (21, 30, 22, 0)

MIN_MAINTENANCE_INTERVAL_HOURS = 72
DEFAULT_MAINTENANCE_INTERVAL_HOURS = 168
RELIABILITY_THRESHOLD = 0.70
URGENCY_RED_THRESHOLD_HOURS = 48
URGENCY_YELLOW_THRESHOLD_HOURS = 168


def extract_failure_intervals(df: pd.DataFrame, device_id: str) -> np.ndarray:
    device_df = df[
        (df['设备编号'] == device_id) &
        (df['记录类型'] == '停机') &
        (df['停机原因分类'] == '设备故障')
    ].copy()

    if len(device_df) == 0:
        return np.array([])

    device_df = device_df.sort_values('开始时间戳').reset_index(drop=True)

    intervals = []
    for i in range(1, len(device_df)):
        prev_end = device_df.loc[i - 1, '结束时间戳']
        curr_start = device_df.loc[i, '开始时间戳']
        interval_hours = (curr_start - prev_end).total_seconds() / 3600.0
        if interval_hours > 0:
            intervals.append(interval_hours)

    return np.array(intervals)


def weibull_pdf(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    x = np.array(x, dtype=float)
    result = np.zeros_like(x)
    mask = x > 0
    x_valid = x[mask]
    result[mask] = (beta / eta) * (x_valid / eta) ** (beta - 1) * \
                   np.exp(-(x_valid / eta) ** beta)
    return result


def weibull_cdf(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    x = np.array(x, dtype=float)
    result = np.zeros_like(x)
    mask = x > 0
    x_valid = x[mask]
    result[mask] = 1 - np.exp(-(x_valid / eta) ** beta)
    return result


def weibull_reliability(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    return 1 - weibull_cdf(x, beta, eta)


def conditional_reliability(t: float, delta_t: float, beta: float, eta: float) -> float:
    if t < 0 or delta_t < 0:
        return 1.0
    R_t = weibull_reliability(np.array([t]), beta, eta)[0]
    R_tdt = weibull_reliability(np.array([t + delta_t]), beta, eta)[0]
    if R_t <= 0:
        return 0.0
    return R_tdt / R_t


def weibull_neg_log_likelihood(params: np.ndarray, data: np.ndarray) -> float:
    beta, eta = params
    if beta <= 0 or eta <= 0:
        return 1e10
    n = len(data)
    if n == 0:
        return 1e10
    log_likelihood = (n * np.log(beta / eta) +
                      (beta - 1) * np.sum(np.log(data / eta)) -
                      np.sum((data / eta) ** beta))
    return -log_likelihood


def fit_weibull_mle(data: np.ndarray) -> Tuple[float, float, bool]:
    if len(data) < 3:
        return 2.0, np.mean(data) if len(data) > 0 else 168.0, False

    try:
        median_init = np.median(data)
        mean_init = np.mean(data)
        eta_init = max(median_init, mean_init, 1.0)
        beta_init = 2.0

        result = minimize(
            weibull_neg_log_likelihood,
            x0=np.array([beta_init, eta_init]),
            args=(data,),
            method='Nelder-Mead',
            bounds=[(0.1, 10.0), (0.1, None)],
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
        )

        if result.success:
            beta, eta = result.x
            if beta > 0 and eta > 0:
                return float(beta), float(eta), True
    except Exception:
        pass

    try:
        x_bar = np.mean(data)
        s = np.std(data, ddof=1) if len(data) > 1 else x_bar * 0.1
        if x_bar > 0 and s > 0:
            cv = s / x_bar
            beta_mm = (1.0 / cv) ** 1.086
            eta_mm = x_bar / gamma(1 + 1 / beta_mm)
            if beta_mm > 0 and eta_mm > 0 and np.isfinite(beta_mm) and np.isfinite(eta_mm):
                return float(max(0.1, min(10.0, beta_mm))), float(max(0.1, eta_mm)), False
    except Exception:
        pass

    return 2.0, float(np.mean(data)) if len(data) > 0 else 168.0, False


def get_last_failure_time(df: pd.DataFrame, device_id: str) -> Optional[datetime]:
    device_df = df[
        (df['设备编号'] == device_id) &
        (df['记录类型'] == '停机') &
        (df['停机原因分类'] == '设备故障')
    ].copy()

    if len(device_df) == 0:
        return None

    return device_df['结束时间戳'].max()


def get_current_time(df: pd.DataFrame) -> datetime:
    if '结束时间戳' in df.columns and len(df) > 0:
        return df['结束时间戳'].max()
    return datetime.now()


def find_delta_t_for_reliability(
    t: float,
    beta: float,
    eta: float,
    target_R: float = 0.70,
    max_delta_t: float = 8760
) -> float:
    if conditional_reliability(t, 0, beta, eta) < target_R:
        return 0.0
    if conditional_reliability(t, max_delta_t, beta, eta) >= target_R:
        return max_delta_t

    low, high = 0.0, max_delta_t
    for _ in range(100):
        mid = (low + high) / 2
        R_mid = conditional_reliability(t, mid, beta, eta)
        if R_mid > target_R:
            low = mid
        else:
            high = mid
        if abs(R_mid - target_R) < 1e-6:
            break
    return (low + high) / 2


def get_urgency_level(delta_t_hours: float, health_score: int, force_urgent: bool = False) -> str:
    if force_urgent or health_score < 50 or delta_t_hours <= URGENCY_RED_THRESHOLD_HOURS:
        return '紧急'
    elif delta_t_hours <= URGENCY_YELLOW_THRESHOLD_HOURS:
        return '临近'
    else:
        return '充裕'


def get_urgency_color(urgency: str) -> str:
    color_map = {
        '紧急': '#dc3545',
        '临近': '#fd7e14',
        '充裕': '#198754',
        '数据不足': '#6c757d',
    }
    return color_map.get(urgency, '#6c757d')


def generate_maintenance_slots(
    start_date: datetime,
    days: int = 7
) -> List[Dict]:
    slots = []
    current_day = datetime(start_date.year, start_date.month, start_date.day)

    for day_offset in range(days):
        slot_day = current_day + timedelta(days=day_offset)

        morning_start = slot_day.replace(
            hour=MORNING_MAINTENANCE_WINDOW[0],
            minute=MORNING_MAINTENANCE_WINDOW[1]
        )
        morning_end = slot_day.replace(
            hour=MORNING_MAINTENANCE_WINDOW[2],
            minute=MORNING_MAINTENANCE_WINDOW[3]
        )
        slots.append({
            'start': morning_start,
            'end': morning_end,
            'shift': '早',
            'type': '班前维护',
        })

        evening_start = slot_day.replace(
            hour=EVENING_MAINTENANCE_WINDOW[0],
            minute=EVENING_MAINTENANCE_WINDOW[1]
        )
        evening_end = slot_day.replace(
            hour=EVENING_MAINTENANCE_WINDOW[2],
            minute=EVENING_MAINTENANCE_WINDOW[3]
        )
        slots.append({
            'start': evening_start,
            'end': evening_end,
            'shift': '晚',
            'type': '班后维护',
        })

    return slots


def calculate_device_maintenance_info(
    df: pd.DataFrame,
    device_id: str,
    health_score: int
) -> Dict:
    current_time = get_current_time(df)
    last_failure = get_last_failure_time(df, device_id)
    intervals = extract_failure_intervals(df, device_id)
    failure_count = len(intervals) + 1 if len(intervals) > 0 else (1 if last_failure else 0)

    result = {
        '设备编号': device_id,
        '健康分': health_score,
        '故障次数': failure_count,
        '故障间隔数': len(intervals),
        '数据充足': len(intervals) >= 3,
        'beta': None,
        'eta': None,
        '拟合成功': False,
        '距上次故障小时': None,
        '当前可靠度': None,
        '建议窗口Δt小时': None,
        '建议维护时间': None,
        '紧迫度': None,
        '使用默认周期': False,
        '强制优先': health_score < 50,
    }

    t = 0.0
    if last_failure:
        t = (current_time - last_failure).total_seconds() / 3600.0
        result['距上次故障小时'] = t

    if len(intervals) >= 3:
        beta, eta, fit_success = fit_weibull_mle(intervals)
        result['beta'] = beta
        result['eta'] = eta
        result['拟合成功'] = fit_success

        if t >= 0:
            current_R = weibull_reliability(np.array([t]), beta, eta)[0]
            result['当前可靠度'] = current_R

            delta_t = find_delta_t_for_reliability(t, beta, eta, RELIABILITY_THRESHOLD)
            result['建议窗口Δt小时'] = delta_t

            suggested_time = current_time + timedelta(hours=delta_t)
            result['建议维护时间'] = suggested_time
    else:
        result['使用默认周期'] = True
        delta_t = DEFAULT_MAINTENANCE_INTERVAL_HOURS
        if t is not None:
            remaining = max(0, DEFAULT_MAINTENANCE_INTERVAL_HOURS - t)
            delta_t = remaining
        result['建议窗口Δt小时'] = delta_t
        result['建议维护时间'] = current_time + timedelta(hours=delta_t)

    if result['使用默认周期']:
        urgency = '数据不足'
    else:
        urgency = get_urgency_level(
            result['建议窗口Δt小时'],
            health_score,
            force_urgent=result['强制优先']
        )
    result['紧迫度'] = urgency

    return result


def assign_maintenance_slots(
    device_infos: List[Dict],
    current_time: datetime,
    horizon_days: int = 7,
    existing_assignments: Optional[List[Dict]] = None
) -> List[Dict]:
    slots = generate_maintenance_slots(current_time, horizon_days)

    device_last_maintenance = {}
    if existing_assignments:
        for a in existing_assignments:
            dev = a['设备编号']
            slot_end = a['维护结束时间']
            if dev not in device_last_maintenance or slot_end > device_last_maintenance[dev]:
                device_last_maintenance[dev] = slot_end

    shift_slot_usage = {}

    priority_queue = []
    for info in device_infos:
        priority_queue.append(info)

    priority_queue.sort(key=lambda x: (
        0 if x['强制优先'] else 1,
        x['建议窗口Δt小时'] if x['建议窗口Δt小时'] is not None else float('inf')
    ))

    assignments = []
    assigned_devices = set()

    for info in priority_queue:
        device_id = info['设备编号']
        if device_id in assigned_devices:
            continue

        suggested_time = info['建议维护时间']
        last_maint = device_last_maintenance.get(device_id)

        best_slot = None
        best_slot_diff = float('inf')

        for slot in slots:
            slot_date_str = slot['start'].strftime('%Y-%m-%d')
            shift_key = (slot_date_str, slot['shift'])
            usage_count = shift_slot_usage.get(shift_key, 0)

            if usage_count >= 1:
                continue

            if last_maint:
                hours_since_last = (slot['start'] - last_maint).total_seconds() / 3600.0
                if hours_since_last < MIN_MAINTENANCE_INTERVAL_HOURS:
                    continue

            if suggested_time:
                time_diff = abs((slot['start'] - suggested_time).total_seconds() / 3600.0)
            else:
                time_diff = 0

            if time_diff < best_slot_diff:
                best_slot_diff = time_diff
                best_slot = slot

        if best_slot is None:
            continue

        shift_key = (best_slot['start'].strftime('%Y-%m-%d'), best_slot['shift'])
        shift_slot_usage[shift_key] = shift_slot_usage.get(shift_key, 0) + 1

        assignment = {
            '设备编号': device_id,
            '维护开始时间': best_slot['start'],
            '维护结束时间': best_slot['end'],
            '维护时长分钟': 30,
            '班次': best_slot['shift'],
            '时段类型': best_slot['type'],
            '紧迫度': info['紧迫度'],
            '威布尔beta': info['beta'],
            '威布尔eta': info['eta'],
            '当前可靠度': info['当前可靠度'],
            '建议窗口Δt小时': info['建议窗口Δt小时'],
            '健康分': info['健康分'],
            '数据充足': info['数据充足'],
            '使用默认周期': info['使用默认周期'],
            '强制优先': info['强制优先'],
            '分配备注': '强制优先分配' if info['强制优先'] else ('按建议窗口分配' if not info['使用默认周期'] else '固定周期分配'),
        }
        assignments.append(assignment)
        assigned_devices.add(device_id)
        device_last_maintenance[device_id] = best_slot['end']

    unassigned = []
    for info in device_infos:
        if info['设备编号'] not in assigned_devices:
            unassigned.append({
                '设备编号': info['设备编号'],
                '原因': '未来7天内可用维护时段已满',
                '建议窗口Δt小时': info['建议窗口Δt小时'],
                '紧迫度': info['紧迫度'],
            })

    return assignments, unassigned


def generate_full_maintenance_schedule(
    df: pd.DataFrame,
    health_scores: Dict[str, Dict],
    horizon_days: int = 7
) -> Dict:
    if df is None or len(df) == 0:
        return {
            '设备信息': [],
            '排程结果': [],
            '未分配设备': [],
            '统计信息': {
                '总设备数': 0,
                '已分配': 0,
                '未分配': 0,
                '数据充足': 0,
                '数据不足': 0,
                '紧急设备': 0,
            }
        }

    current_time = get_current_time(df)
    device_list = sorted(df['设备编号'].unique().tolist())

    device_infos = []
    for device_id in device_list:
        hs = 0
        if device_id in health_scores:
            hs = health_scores[device_id]['健康分']
        info = calculate_device_maintenance_info(df, device_id, hs)
        device_infos.append(info)

    assignments, unassigned = assign_maintenance_slots(
        device_infos,
        current_time,
        horizon_days
    )

    stats = {
        '总设备数': len(device_list),
        '已分配': len(assignments),
        '未分配': len(unassigned),
        '数据充足': sum(1 for i in device_infos if i['数据充足']),
        '数据不足': sum(1 for i in device_infos if not i['数据充足']),
        '紧急设备': sum(1 for i in device_infos if i['紧迫度'] == '紧急'),
    }

    def _to_py_datetime(obj):
        if obj is None:
            return None
        if hasattr(obj, 'to_pydatetime'):
            return obj.to_pydatetime()
        if isinstance(obj, dict):
            return {k: _to_py_datetime(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_py_datetime(v) for v in obj]
        return obj

    return {
        '设备信息': _to_py_datetime(device_infos),
        '排程结果': _to_py_datetime(assignments),
        '未分配设备': _to_py_datetime(unassigned),
        '统计信息': stats,
        '当前时间': _to_py_datetime(current_time),
    }
