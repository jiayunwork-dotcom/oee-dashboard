import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from src.oee_engine import calculate_oee_overall, downtime_by_category
from src.trend_analysis import calculate_overall_daily_oee, get_all_anomalies


def create_matplotlib_pareto(downtime_df: pd.DataFrame) -> bytes:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    if len(downtime_df) == 0:
        ax1.set_title('暂无停机数据')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.read()
    
    categories = downtime_df['停机原因'].tolist()
    values = downtime_df['持续时间分钟'].tolist()
    
    cumulative = np.cumsum(values) / sum(values) * 100
    
    bars = ax1.bar(categories, values, color='#3498db')
    ax1.set_ylabel('停机时间(分钟)')
    ax1.set_xlabel('停机原因')
    ax1.set_title('停机原因Pareto分析')
    
    ax2 = ax1.twinx()
    ax2.plot(categories, cumulative, 'ro-', linewidth=2, markersize=6)
    ax2.set_ylabel('累积百分比(%)')
    ax2.set_ylim(0, 105)
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% 分界线')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def create_matplotlib_trend(daily_df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if len(daily_df) == 0:
        ax.set_title('暂无趋势数据')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.read()
    
    dates = pd.to_datetime(daily_df['日期'])
    
    ax.plot(dates, daily_df['OEE'] * 100, color='blue', marker='o', 
            linestyle='-', linewidth=2, label='OEE', markersize=4)
    ax.plot(dates, daily_df['可用率'] * 100, color='green', marker='s', 
            linestyle='--', linewidth=1.5, label='可用率', markersize=3)
    ax.plot(dates, daily_df['性能率'] * 100, color='orange', marker='^', 
            linestyle='--', linewidth=1.5, label='性能率', markersize=3)
    ax.plot(dates, daily_df['质量率'] * 100, color='purple', marker='d', 
            linestyle='--', linewidth=1.5, label='质量率', markersize=3)
    
    ax.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='目标值 85%')
    
    ax.set_ylabel('效率 (%)')
    ax.set_xlabel('日期')
    ax.set_title('OEE趋势图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def create_matplotlib_gantt(df: pd.DataFrame, devices: List[str]) -> bytes:
    fig, ax = plt.subplots(figsize=(12, max(4, len(devices) * 0.8)))
    
    color_map = {
        '运行': '#2ecc71',
        '设备故障': '#e74c3c',
        '换模调整': '#f39c12',
        '空转短停': '#f1c40f',
        '计划维护': '#95a5a6',
    }
    
    for idx, device in enumerate(reversed(devices)):
        device_data = df[df['设备编号'] == device].sort_values('开始时间戳')
        
        for _, row in device_data.iterrows():
            record_type = row['记录类型']
            if record_type == '停机':
                reason = row['停机原因分类']
                color = color_map.get(reason, '#e74c3c') if reason else '#e74c3c'
            else:
                type_map = {
                    '运行': '运行',
                    '换模': '换模调整',
                    '空转': '空转短停',
                }
                display_type = type_map.get(record_type, record_type)
                color = color_map.get(display_type, '#3498db')
            
            start = row['开始时间戳']
            end = row['结束时间戳']
            duration = (end - start).total_seconds() / 3600
            
            ax.barh(idx, duration, left=start.hour + start.minute/60, 
                    height=0.6, color=color, edgecolor='none')
    
    ax.set_yticks(range(len(devices)))
    ax.set_yticklabels(list(reversed(devices)))
    ax.set_xlabel('时间 (小时)')
    ax.set_title('设备运行甘特图')
    ax.set_xlim(0, 24)
    ax.grid(True, axis='x', alpha=0.3)
    
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label)
        for label, color in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def generate_suggestions(pareto_df: pd.DataFrame) -> List[str]:
    suggestions = {
        '设备故障': [
            '建议建立设备预防性维护计划，定期检查关键部件',
            '分析高频故障模式，建立备件库存策略',
            '开展设备操作员技能培训，减少误操作导致的故障',
        ],
        '换模调整': [
            '推广SMED(快速换模)方法，目标将换模时间缩短50%',
            '建立换模标准作业流程，减少换模调整波动',
            '准备换模工具车，减少工具寻找时间',
        ],
        '空转短停': [
            '开展全员生产维护(TPM)，减少微缺陷积累',
            '优化物料配送，减少缺料等待时间',
            '分析短停原因，制定针对性改善措施',
        ],
        '计划维护': [
            '优化维护计划，尽量安排在非生产时段',
            '评估维护内容的必要性，减少过度维护',
            '推进预测性维护，提高维护效率',
        ],
        '缺料': [
            '优化供应链管理，建立安全库存机制',
            '加强供应商质量管理，减少来料不良',
            '建立物料预警系统，提前发现缺料风险',
        ],
        '质量': [
            '分析缺陷类型，找出根本原因',
            '推进质量改进项目，提高过程能力',
            '加强首件检验和过程检验，减少批量不良',
        ],
    }
    
    all_suggestions = []
    for _, row in pareto_df.head(3).iterrows():
        reason = row['停机原因']
        if reason in suggestions:
            all_suggestions.append(f"【{reason}】:")
            all_suggestions.extend(suggestions[reason])
    
    return all_suggestions


def generate_pdf_report(df: pd.DataFrame, start_date: str, end_date: str,
                         report_type: str = '日报',
                         configured_takts: Optional[Dict[str, float]] = None,
                         oee_target: float = 0.85) -> bytes:
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.darkblue,
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    story = []
    
    story.append(Paragraph(f'OEE分析报告 - {report_type}', title_style))
    story.append(Paragraph(f'报告期间: {start_date} 至 {end_date}', ParagraphStyle(
        'SubTitle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, spaceAfter=30,
        textColor=colors.grey,
    )))
    
    overall = calculate_oee_overall(df, start_date, end_date, configured_takts)
    summary = overall['设备汇总']
    
    story.append(Paragraph('一、总览', h2_style))
    
    summary_data = [
        ['指标', '数值', '目标值', '达成状态'],
        ['OEE', f"{summary['OEE']*100:.2f}%", f"{oee_target*100:.0f}%", 
         '✓ 达标' if summary['OEE'] >= oee_target else '✗ 未达标'],
        ['可用率', f"{summary['可用率']*100:.2f}%", '90%', 
         '✓ 达标' if summary['可用率'] >= 0.9 else '✗ 未达标'],
        ['性能率', f"{summary['性能率']*100:.2f}%", '95%', 
         '✓ 达标' if summary['性能率'] >= 0.95 else '✗ 未达标'],
        ['质量率', f"{summary['质量率']*100:.2f}%", '99.9%', 
         '✓ 达标' if summary['质量率'] >= 0.999 else '✗ 未达标'],
    ]
    
    summary_table = Table(summary_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(summary_table)
    
    story.append(Paragraph('二、趋势分析', h2_style))
    daily_oee = calculate_overall_daily_oee(df, start_date, end_date, configured_takts)
    trend_img = create_matplotlib_trend(daily_oee)
    trend_image = Image(io.BytesIO(trend_img), width=16*cm, height=8*cm)
    story.append(trend_image)
    
    story.append(Paragraph('三、停机原因Pareto分析', h2_style))
    pareto_df = downtime_by_category(df, start_date, end_date)
    pareto_img = create_matplotlib_pareto(pareto_df)
    pareto_image = Image(io.BytesIO(pareto_img), width=16*cm, height=8*cm)
    story.append(pareto_image)
    
    if len(pareto_df) > 0:
        story.append(Paragraph('停机原因明细:', styles['Heading3']))
        detail_data = [['排名', '停机原因', '停机时间(分钟)', '占比', '累积占比']]
        total = pareto_df['持续时间分钟'].sum()
        cumulative = 0
        for idx, (_, row) in enumerate(pareto_df.iterrows(), 1):
            cumulative += row['持续时间分钟']
            detail_data.append([
                str(idx),
                row['停机原因'],
                f"{row['持续时间分钟']:.1f}",
                f"{row['持续时间分钟']/total*100:.1f}%",
                f"{cumulative/total*100:.1f}%",
            ])
        
        detail_table = Table(detail_data, colWidths=[2*cm, 4*cm, 3.5*cm, 2.5*cm, 3*cm])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(detail_table)
    
    story.append(PageBreak())
    story.append(Paragraph('四、异常事件清单', h2_style))
    
    anomalies = get_all_anomalies(df, configured_takts, oee_target)
    if anomalies:
        anomaly_data = [['序号', '类型', '日期', '设备', '详情']]
        for idx, a in enumerate(anomalies[:10], 1):
            if a['类型'] == 'OEE骤降告警':
                detail = f"OEE从{a['7日均值']*100:.1f}%下降至{a['当前OEE']*100:.1f}%，下降{a['下降幅度']*100:.1f}个百分点，主要原因为{a['主要归因因子']}"
            else:
                detail = f"连续{a['持续天数']}天低于目标值，平均OEE {a['平均OEE']*100:.1f}%"
            
            anomaly_data.append([
                str(idx),
                a['类型'],
                str(a.get('日期', '')),
                a['设备'],
                detail,
            ])
        
        anomaly_table = Table(anomaly_data, colWidths=[1.5*cm, 2.5*cm, 2.5*cm, 2*cm, 8*cm])
        anomaly_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightsalmon),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        story.append(anomaly_table)
    else:
        story.append(Paragraph('本期无异常事件', normal_style))
    
    story.append(Paragraph('五、改善建议', h2_style))
    suggestions = generate_suggestions(pareto_df)
    if suggestions:
        for suggestion in suggestions:
            story.append(Paragraph(f'• {suggestion}', normal_style))
            story.append(Spacer(1, 5))
    
    story.append(Spacer(1, 30))
    story.append(Paragraph(f'报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                           ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                                          textColor=colors.grey, alignment=TA_LEFT)))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
