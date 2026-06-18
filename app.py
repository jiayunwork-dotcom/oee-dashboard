import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, date, timedelta
import json

from src.data_import import validate_csv, process_dataframe, load_csv, get_device_list, get_date_range
from src.oee_engine import calculate_oee_overall, calculate_oee_for_device, downtime_by_category
from src.downtime_analysis import create_pareto_chart, create_gantt_chart, create_downtime_pie, create_root_cause_drilldown
from src.trend_analysis import (
    create_trend_chart, create_factor_trend_chart, 
    create_device_comparison_chart, create_shift_comparison_chart,
    create_benchmark_chart, get_all_anomalies, calculate_overall_daily_oee
)
from src.report_export import generate_pdf_report
from src.health_score import (
    calculate_all_health_scores, calculate_health_score_trend,
    calculate_health_score_for_device,
    get_health_level, detect_health_score_drop_anomalies
)
from src.predictive_maintenance import (
    generate_full_maintenance_schedule, get_urgency_color,
    get_urgency_level
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = 'OEE设备综合效率分析系统'
server = app.server

DATA_STORE = {
    'df': None,
    'processed_df': None,
}

DEFAULT_TAKT = {
    'CNC-001': 45,
    'CNC-002': 50,
    'CNC-003': 40,
    '冲压-001': 15,
    '焊接-001': 60,
}


def get_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("🏭 OEE设备综合效率分析系统", className="ms-2", style={"fontSize": "1.25rem"}),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("📊 总览", href="/", active="exact")),
                    dbc.NavItem(dbc.NavLink("📥 数据导入", href="/data-import", active="exact")),
                    dbc.NavItem(dbc.NavLink("📈 趋势分析", href="/trend-analysis", active="exact")),
                    dbc.NavItem(dbc.NavLink("🔧 停机归因", href="/downtime-analysis", active="exact")),
                    dbc.NavItem(dbc.NavLink("🏆 对标分析", href="/benchmark", active="exact")),
                    dbc.NavItem(dbc.NavLink("⚠️ 异常检测", href="/anomaly", active="exact")),
                    dbc.NavItem(dbc.NavLink("🔧 维保排程", href="/maintenance-schedule", active="exact")),
                    dbc.NavItem(dbc.NavLink("📄 报告导出", href="/report", active="exact")),
                ],
                pills=True,
            ),
        ]),
        color="primary",
        dark=True,
        sticky="top",
    )


def get_footer():
    return html.Footer(
        dbc.Container(
            html.P("© 2025 OEE设备综合效率分析系统 - 制造业智能制造解决方案", 
                   className="text-center text-muted mt-4 mb-2"),
        ),
        style={"backgroundColor": "#f8f9fa", "padding": "1rem 0"},
    )


def get_date_range_picker():
    return dbc.Row([
        dbc.Col([
            html.Label("开始日期", className="form-label"),
            dcc.DatePickerSingle(
                id='start-date-picker',
                display_format='YYYY-MM-DD',
                className="form-control",
            ),
        ], width=3),
        dbc.Col([
            html.Label("结束日期", className="form-label"),
            dcc.DatePickerSingle(
                id='end-date-picker',
                display_format='YYYY-MM-DD',
                className="form-control",
            ),
        ], width=3),
        dbc.Col([
            html.Label("选择设备", className="form-label"),
            dcc.Dropdown(
                id='device-dropdown',
                multi=True,
                placeholder="选择设备（多选）",
                className="form-select",
            ),
        ], width=6),
    ], className="mb-3")


def get_kpi_card(title, value, subtitle="", color="primary", icon="📊"):
    color_map = {
        "primary": "#0d6efd",
        "success": "#198754",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "info": "#0dcaf0",
    }
    bg_color = color_map.get(color, "#0d6efd")
    
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={"fontSize": "2rem", "marginRight": "1rem"}),
                html.Div([
                    html.H5(title, className="card-title mb-1", style={"color": "#6c757d"}),
                    html.H3(value, className="card-text mb-0", style={"color": bg_color, "fontWeight": "bold"}),
                    html.Small(subtitle, className="text-muted") if subtitle else None,
                ]),
            ], style={"display": "flex", "alignItems": "center"}),
        ]),
        className="shadow-sm",
    )


def get_overview_layout():
    return dbc.Container([
        html.H2("📊 OEE总览", className="mb-4 mt-3"),
        
        get_date_range_picker(),
        
        html.Hr(),
        
        dbc.Row([
            dbc.Col(get_kpi_card("整体OEE", "--", "目标: 85%", "primary", "🎯"), width=3),
            dbc.Col(get_kpi_card("可用率", "--", "目标: 90%", "success", "⚙️"), width=3),
            dbc.Col(get_kpi_card("性能率", "--", "目标: 95%", "warning", "⚡"), width=3),
            dbc.Col(get_kpi_card("质量率", "--", "目标: 99.9%", "info", "✅"), width=3),
        ], className="mb-4", id="kpi-cards"),
        
        dbc.Card([
            dbc.CardHeader([
                html.H5("💚 设备健康概览", className="mb-0"),
                html.Small(" 基于最近7天数据计算，点击卡片跳转到停机归因分析", className="text-muted ms-2"),
            ]),
            dbc.CardBody(id='health-score-cards'),
        ], className="shadow-sm mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("OEE三因子构成"),
                    dbc.CardBody([
                        dcc.Graph(id='oee-breakdown-chart'),
                    ]),
                ], className="shadow-sm"),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("各设备OEE排名"),
                    dbc.CardBody([
                        dcc.Graph(id='device-oee-ranking'),
                    ]),
                ], className="shadow-sm"),
            ], width=6),
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader(html.Div([
                "🔍 根因钻取分析",
                html.Small(" (点击下方各层级逐步下钻定位问题)", className="text-muted ms-2"),
            ])),
            dbc.CardBody(id='drilldown-panel'),
        ], className="shadow-sm mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("设备运行甘特图"),
                    dbc.CardBody([
                        dcc.Graph(id='overview-gantt-chart'),
                    ]),
                ], className="shadow-sm"),
            ], width=12),
        ], className="mb-4"),
        
    ], fluid=True)


def get_data_import_layout():
    return dbc.Container([
        html.H2("📥 数据导入", className="mb-4 mt-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("上传CSV文件"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                '拖拽CSV文件到此处 或 ',
                                html.A('点击选择文件'),
                            ]),
                            style={
                                'width': '100%',
                                'height': '120px',
                                'lineHeight': '120px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'margin': '10px 0',
                                'backgroundColor': '#f8f9fa',
                            },
                            multiple=False,
                        ),
                        html.Div(id='upload-status'),
                    ]),
                ], className="shadow-sm mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("或使用示例数据"),
                    dbc.CardBody([
                        dbc.Button("加载示例数据", id='load-sample-btn', color="secondary", className="me-2"),
                        html.Small(" 快速体验系统功能", className="text-muted"),
                    ]),
                ], className="shadow-sm mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("数据校验结果"),
                    dbc.CardBody([
                        html.Div(id='validation-results'),
                    ]),
                ], className="shadow-sm mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("💚 健康评分预览"),
                    dbc.CardBody([
                        html.Div(id='health-score-preview'),
                    ]),
                ], className="shadow-sm"),
                
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("数据预览"),
                    dbc.CardBody([
                        html.Div(id='data-preview'),
                    ]),
                ], className="shadow-sm mt-4"),
            ], width=12),
        ]),
        
    ], fluid=True)


def get_trend_analysis_layout():
    return dbc.Container([
        html.H2("📈 趋势分析", className="mb-4 mt-3"),
        
        get_date_range_picker(),
        
        dbc.Tabs([
            dbc.Tab(label="OEE趋势", tab_id="oee-trend"),
            dbc.Tab(label="健康分趋势", tab_id="health-trend"),
        ], id="trend-tabs", active_tab="oee-trend", className="mb-4"),
        
        html.Div(id='oee-trend-content', style={'display': 'block'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("趋势类型"),
                        dbc.CardBody([
                            dbc.RadioItems(
                                id='trend-type',
                                options=[
                                    {'label': '日趋势', 'value': '日'},
                                    {'label': '周趋势', 'value': '周'},
                                    {'label': '月趋势', 'value': '月'},
                                ],
                                value='日',
                                inline=True,
                            ),
                        ]),
                    ], className="shadow-sm"),
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("显示设置"),
                        dbc.CardBody([
                            dbc.Checklist(
                                id='trend-options',
                                options=[
                                    {'label': '显示7日移动平均线', 'value': 'moving_avg'},
                                    {'label': '显示三因子趋势', 'value': 'show_factors'},
                                ],
                                value=['moving_avg'],
                                inline=True,
                            ),
                        ]),
                    ], className="shadow-sm"),
                ], width=8),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("OEE趋势图"),
                        dbc.CardBody([
                            dcc.Graph(id='trend-chart'),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("单设备三因子趋势"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='factor-trend-device',
                                placeholder="选择设备查看三因子趋势",
                                className="mb-3",
                            ),
                            dcc.Graph(id='factor-trend-chart'),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
        ]),
        
        html.Div(id='health-trend-content', style={'display': 'none'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("设备选择"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='health-trend-devices',
                                multi=True,
                                placeholder="选择设备（最多5台）",
                                className="mb-2",
                            ),
                            html.Small("支持多设备叠加对比，最多选择5台设备", className="text-muted"),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            "健康分趋势图",
                            html.Small(" （基于7天滑动窗口计算）", className="text-muted ms-2"),
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='health-trend-chart'),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("等级说明"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Span(" ", style={
                                            'display': 'inline-block',
                                            'width': '20px',
                                            'height': '20px',
                                            'backgroundColor': '#198754',
                                            'borderRadius': '3px',
                                            'marginRight': '8px',
                                            'verticalAlign': 'middle',
                                        }),
                                        html.Strong("优秀 (90-100分)"),
                                    ]),
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Span(" ", style={
                                            'display': 'inline-block',
                                            'width': '20px',
                                            'height': '20px',
                                            'backgroundColor': '#0d6efd',
                                            'borderRadius': '3px',
                                            'marginRight': '8px',
                                            'verticalAlign': 'middle',
                                        }),
                                        html.Strong("良好 (70-89分)"),
                                    ]),
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Span(" ", style={
                                            'display': 'inline-block',
                                            'width': '20px',
                                            'height': '20px',
                                            'backgroundColor': '#fd7e14',
                                            'borderRadius': '3px',
                                            'marginRight': '8px',
                                            'verticalAlign': 'middle',
                                        }),
                                        html.Strong("关注 (50-69分)"),
                                    ]),
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Span(" ", style={
                                            'display': 'inline-block',
                                            'width': '20px',
                                            'height': '20px',
                                            'backgroundColor': '#dc3545',
                                            'borderRadius': '3px',
                                            'marginRight': '8px',
                                            'verticalAlign': 'middle',
                                        }),
                                        html.Strong("警告 (0-49分)"),
                                    ]),
                                ], width=3),
                            ]),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
        ]),
        
    ], fluid=True)


def get_downtime_analysis_layout():
    return dbc.Container([
        html.H2("🔧 停机归因分析", className="mb-4 mt-3"),
        
        get_date_range_picker(),
        
        dbc.Tabs([
            dbc.Tab(label="Pareto分析", tab_id="pareto"),
            dbc.Tab(label="时间轴甘特图", tab_id="gantt"),
            dbc.Tab(label="停机原因分布", tab_id="pie"),
            dbc.Tab(label="根因钻取", tab_id="drilldown"),
        ], id="downtime-tabs", active_tab="pareto", className="mb-4"),
        
        html.Div(id='downtime-tab-content'),
        
    ], fluid=True)


def get_benchmark_layout():
    return dbc.Container([
        html.H2("🏆 对标分析", className="mb-4 mt-3"),
        
        get_date_range_picker(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("与世界级标杆对比"),
                    dbc.CardBody([
                        dcc.Graph(id='benchmark-chart'),
                    ]),
                ], className="shadow-sm"),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("各设备OEE排名"),
                    dbc.CardBody([
                        dcc.Graph(id='benchmark-device-ranking'),
                    ]),
                ], className="shadow-sm"),
            ], width=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("班次对比分析"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='shift-compare-device',
                            placeholder="选择设备查看班次对比",
                            className="mb-3",
                        ),
                        dcc.Graph(id='shift-comparison-chart'),
                    ]),
                ], className="shadow-sm"),
            ], width=12),
        ], className="mb-4"),
        
    ], fluid=True)


def get_anomaly_layout():
    return dbc.Container([
        html.H2("⚠️ 异常检测", className="mb-4 mt-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("异常检测参数"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("OEE目标值", className="form-label"),
                                dbc.Input(
                                    id='anomaly-oee-target',
                                    type='number',
                                    value=0.85,
                                    min=0,
                                    max=1,
                                    step=0.01,
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("骤降阈值(百分点)", className="form-label"),
                                dbc.Input(
                                    id='anomaly-drop-threshold',
                                    type='number',
                                    value=15,
                                    min=1,
                                    max=100,
                                    step=1,
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("连续低效天数", className="form-label"),
                                dbc.Input(
                                    id='anomaly-consecutive-days',
                                    type='number',
                                    value=3,
                                    min=1,
                                    max=30,
                                    step=1,
                                ),
                            ], width=4),
                        ]),
                    ]),
                ], className="shadow-sm mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("异常事件列表"),
                    dbc.CardBody([
                        html.Div(id='anomaly-cards'),
                    ]),
                ], className="shadow-sm"),
            ], width=12),
        ], className="mb-4"),
        
    ], fluid=True)


def get_maintenance_schedule_layout():
    return dbc.Container([
        html.H2("🔧 维保排程优化", className="mb-4 mt-3"),
        
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H5("排程参数设置", className="mb-0"),
                    html.Small(" 基于威布尔分布的预防性维护排程优化", className="text-muted ms-2"),
                ]),
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("排程时间范围(天)", className="form-label"),
                        dbc.Input(
                            id='schedule-horizon',
                            type='number',
                            value=7,
                            min=3,
                            max=30,
                            step=1,
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("可靠度阈值(%)", className="form-label"),
                        dbc.Input(
                            id='reliability-threshold',
                            type='number',
                            value=70,
                            min=50,
                            max=95,
                            step=5,
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("强制维护健康分阈值", className="form-label"),
                        dbc.Input(
                            id='health-threshold',
                            type='number',
                            value=50,
                            min=30,
                            max=80,
                            step=5,
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("&nbsp;", className="form-label d-block"),
                        dbc.Button("📊 生成排程", id='generate-schedule-btn', color="primary", className="w-100"),
                    ], width=3),
                ]),
            ]),
        ], className="shadow-sm mb-4"),
        
        dbc.Row(id='schedule-kpi-cards', className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H5("排程甘特图", className="mb-0"),
                    html.Small(" 横轴: 未来7天维护时段 | 纵轴: 各设备 | 色块颜色表示紧迫度", className="text-muted ms-2"),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span(" ", style={
                                'display': 'inline-block',
                                'width': '20px',
                                'height': '20px',
                                'backgroundColor': '#dc3545',
                                'borderRadius': '3px',
                                'marginRight': '8px',
                                'verticalAlign': 'middle',
                            }),
                            html.Strong("紧急"),
                            html.Span("  (≤48小时或健康分<50)", className="text-muted ms-1 me-4"),
                            html.Span(" ", style={
                                'display': 'inline-block',
                                'width': '20px',
                                'height': '20px',
                                'backgroundColor': '#fd7e14',
                                'borderRadius': '3px',
                                'marginRight': '8px',
                                'verticalAlign': 'middle',
                            }),
                            html.Strong("临近"),
                            html.Span("  (≤7天)", className="text-muted ms-1 me-4"),
                            html.Span(" ", style={
                                'display': 'inline-block',
                                'width': '20px',
                                'height': '20px',
                                'backgroundColor': '#198754',
                                'borderRadius': '3px',
                                'marginRight': '8px',
                                'verticalAlign': 'middle',
                            }),
                            html.Strong("充裕"),
                            html.Span("  (>7天)", className="text-muted ms-1 me-4"),
                            html.Span(" ", style={
                                'display': 'inline-block',
                                'width': '20px',
                                'height': '20px',
                                'backgroundColor': '#6c757d',
                                'borderRadius': '3px',
                                'marginRight': '8px',
                                'verticalAlign': 'middle',
                            }),
                            html.Strong("数据不足"),
                            html.Span("  (故障记录<3次)", className="text-muted ms-1"),
                        ], style={'fontSize': '0.875rem'}),
                    ], width=12),
                ], className="mt-2"),
            ]),
            dbc.CardBody([
                dcc.Graph(id='maintenance-gantt-chart'),
            ]),
        ], className="shadow-sm mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("📋 维护排程列表", className="mb-0"),
                            html.Div([
                                dbc.Button(
                                    "📅 导出iCalendar(.ics)",
                                    id='export-ics-btn',
                                    color="success",
                                    size="sm",
                                ),
                                dcc.Download(id='download-ics'),
                            ]),
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                    ]),
                    dbc.CardBody(id='schedule-list'),
                ], className="shadow-sm"),
            ], width=12),
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader([
                html.H5("📊 设备威布尔参数与可靠度", className="mb-0"),
                html.Small(" 基于历史故障数据拟合威布尔分布(形状参数β, 尺度参数η)", className="text-muted ms-2"),
            ]),
            dbc.CardBody(id='weibull-params-panel'),
        ], className="shadow-sm mb-4"),
        
    ], fluid=True)


def get_report_layout():
    return dbc.Container([
        html.H2("📄 报告导出", className="mb-4 mt-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("报告设置"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("报告类型", className="form-label"),
                                dbc.Select(
                                    id='report-type',
                                    options=[
                                        {'label': '日报', 'value': '日报'},
                                        {'label': '周报', 'value': '周报'},
                                        {'label': '月报', 'value': '月报'},
                                    ],
                                    value='日报',
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label("开始日期", className="form-label"),
                                dcc.DatePickerSingle(
                                    id='report-start-date',
                                    display_format='YYYY-MM-DD',
                                    className="form-control",
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label("结束日期", className="form-label"),
                                dcc.DatePickerSingle(
                                    id='report-end-date',
                                    display_format='YYYY-MM-DD',
                                    className="form-control",
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label("&nbsp;", className="form-label d-block"),
                                dbc.Button("生成PDF报告", id='generate-report-btn', color="primary"),
                            ], width=3),
                        ]),
                        html.Div(id='report-download-area', className="mt-3"),
                    ]),
                ], className="shadow-sm mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("报告预览"),
                    dbc.CardBody([
                        html.Div(id='report-preview'),
                    ]),
                ], className="shadow-sm"),
            ], width=12),
        ], className="mb-4"),
        
    ], fluid=True)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='data-loaded-store', data=False),
    dcc.Store(id='drilldown-state', data={}),
    dcc.Store(id='previous-health-scores', data={}),
    dcc.Store(id='current-health-scores', data={}),
    dcc.Store(id='url-device-param', data=None),
    get_navbar(),
    html.Div(id='page-content', style={"paddingTop": "70px"}),
    get_footer(),
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/data-import':
        return get_data_import_layout()
    elif pathname == '/trend-analysis':
        return get_trend_analysis_layout()
    elif pathname == '/downtime-analysis':
        return get_downtime_analysis_layout()
    elif pathname == '/benchmark':
        return get_benchmark_layout()
    elif pathname == '/anomaly':
        return get_anomaly_layout()
    elif pathname == '/maintenance-schedule':
        return get_maintenance_schedule_layout()
    elif pathname == '/report':
        return get_report_layout()
    else:
        return get_overview_layout()


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
        else:
            return None, ['请上传CSV格式文件']
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('gbk')))
        except Exception as e:
            return None, [f'文件解码失败: {str(e)}']
    except Exception as e:
        return None, [f'文件读取失败: {str(e)}']
    
    is_valid, errors, validated_df = validate_csv(df)
    return validated_df, errors


@app.callback(
    [Output('upload-status', 'children'),
     Output('validation-results', 'children'),
     Output('data-preview', 'children'),
     Output('data-loaded-store', 'data'),
     Output('health-score-preview', 'children'),
     Output('previous-health-scores', 'data'),
     Output('current-health-scores', 'data')],
    [Input('upload-data', 'contents'),
     Input('load-sample-btn', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('current-health-scores', 'data')]
)
def handle_file_upload(contents, n_clicks, filename, old_current_scores):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return html.Div("请上传数据文件或加载示例数据", className="text-muted"), \
               html.Div("暂无数据", className="text-muted"), \
               html.Div("暂无数据", className="text-muted"), False, \
               html.Div("暂无数据", className="text-muted"), {}, {}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    df = None
    errors = []
    
    if trigger_id == 'load-sample-btn':
        try:
            from src.generate_sample_data import generate_sample_data
            df = generate_sample_data()
            is_valid, errors, validated_df = validate_csv(df)
            df = validated_df
        except Exception as e:
            return dbc.Alert(f"加载示例数据失败: {str(e)}", color="danger"), \
                   html.Div(""), html.Div(""), False, \
                   html.Div(""), {}, {}
        
        upload_status = dbc.Alert("✅ 示例数据加载成功！", color="success")
    elif trigger_id == 'upload-data' and contents is not None:
        df, errors = parse_contents(contents, filename)
        if df is None:
            return dbc.Alert(f"❌ 文件上传失败: {errors[0] if errors else '未知错误'}", color="danger"), \
                   html.Div(""), html.Div(""), False, \
                   html.Div(""), {}, {}
        
        if errors:
            upload_status = dbc.Alert(f"⚠️ 数据校验发现 {len(errors)} 个问题，请修正后重新上传", color="warning")
        else:
            upload_status = dbc.Alert("✅ 数据上传成功，校验通过！", color="success")
    else:
        return html.Div("请上传数据文件或加载示例数据", className="text-muted"), \
               html.Div("暂无数据", className="text-muted"), \
               html.Div("暂无数据", className="text-muted"), False, \
               html.Div("暂无数据", className="text-muted"), {}, {}
    
    if df is not None and len(df) > 0:
        DATA_STORE['df'] = df
        DATA_STORE['processed_df'] = process_dataframe(df)
        
        preview = dash_table.DataTable(
            data=df.head(20).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        )
        data_loaded = True
    else:
        preview = html.Div("暂无数据", className="text-muted")
        data_loaded = False
    
    if errors:
        error_list = html.Div([
            html.H6("数据统计:", className="text-success"),
            html.Ul([
                html.Li(f"记录总数: {len(df)}"),
                html.Li(f"设备数量: {df['设备编号'].nunique()}"),
                html.Li(f"日期范围: {df['日期'].min()} 至 {df['日期'].max()}"),
            ]),
            html.Hr(),
            dbc.Alert(f"⚠️ 发现 {len(errors)} 个校验问题，请修正后重新上传", color="warning", className="mt-3"),
            html.H6("校验错误明细:", style={"color": "#dc3545"}),
            html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("#", style={"width": "5%", "padding": "6px 10px", "backgroundColor": "#f8f9fa", "textAlign": "center"}),
                        html.Th("错误描述", style={"padding": "6px 10px", "backgroundColor": "#f8f9fa"}),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(str(i + 1), style={"padding": "4px 10px", "textAlign": "center", "borderBottom": "1px solid #dee2e6"}),
                            html.Td(e, style={"padding": "4px 10px", "borderBottom": "1px solid #dee2e6", "color": "#dc3545"}),
                        ]) for i, e in enumerate(errors[:50])
                    ]),
                ], style={"width": "100%", "borderCollapse": "collapse", "border": "1px solid #dee2e6", "borderRadius": "6px"}),
            ], style={"maxHeight": "300px", "overflowY": "auto"}),
            html.Small(f"共 {len(errors)} 个错误，显示前{min(len(errors), 50)}个", className="text-muted mt-2") if len(errors) > 50 else None,
        ])
    else:
        error_list = html.Div([
            html.H6("数据统计:", className="text-success"),
            html.Ul([
                html.Li(f"记录总数: {len(df)}"),
                html.Li(f"设备数量: {df['设备编号'].nunique()}"),
                html.Li(f"日期范围: {df['日期'].min()} 至 {df['日期'].max()}"),
            ]),
            dbc.Alert("✅ 所有数据校验通过，无异常", color="success", className="mt-3"),
        ])
    
    health_preview = html.Div("暂无数据", className="text-muted")
    current_scores_dict = {}
    
    if df is not None and len(df) > 0 and not errors:
        processed_df = process_dataframe(df)
        max_date = str(df['日期'].max())
        health_scores = calculate_all_health_scores(processed_df, max_date, DEFAULT_TAKT, 7)
        
        has_insufficient_data = any(not s['数据充足'] for s in health_scores.values())
        
        score_rows = []
        for device_id, score_data in sorted(health_scores.items()):
            score = score_data['健康分']
            level = get_health_level(score)
            data_note = "" if score_data['数据充足'] else " (数据不足7天)"
            
            score_rows.append(html.Tr([
                html.Td(html.Strong(device_id)),
                html.Td([
                    html.Span(
                        f"{score}分",
                        style={
                            'fontWeight': 'bold',
                            'color': level['color'],
                            'fontSize': '1.1rem',
                        }
                    ),
                    html.Span(
                        level['name'],
                        style={
                            'backgroundColor': level['bg_color'],
                            'color': level['color'],
                            'padding': '2px 10px',
                            'borderRadius': '12px',
                            'fontSize': '0.75rem',
                            'marginLeft': '10px',
                        }
                    ),
                    html.Small(data_note, className="text-muted ms-2") if data_note else None,
                ]),
                html.Td(f"{score_data['可用率']*100:.1f}%"),
                html.Td(f"{score_data['性能率']*100:.1f}%"),
                html.Td(f"{score_data['质量率']*100:.1f}%"),
                html.Td(f"-{score_data['稳定性惩罚']:.1f}"),
                html.Td(f"{score_data['数据天数']}天"),
            ]))
        
        health_preview = html.Div([
            html.P("基于导入数据范围内的健康评分计算：", className="text-muted mb-2"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("设备"),
                    html.Th("健康分"),
                    html.Th("可用率"),
                    html.Th("性能率"),
                    html.Th("质量率"),
                    html.Th("稳定性惩罚"),
                    html.Th("数据天数"),
                ])),
                html.Tbody(score_rows),
            ], className="table table-sm table-hover"),
            html.Small("💡 数据不足7天时，评分仅供参考", className="text-muted") if has_insufficient_data else None,
        ])
        
        current_scores_dict = convert_scores_to_serializable({k: v for k, v in health_scores.items()})
    
    new_previous_scores = old_current_scores if old_current_scores else {}
    
    return upload_status, error_list, preview, data_loaded, health_preview, new_previous_scores, current_scores_dict


def convert_scores_to_serializable(scores_dict):
    if not scores_dict:
        return {}
    result = {}
    for device, data in scores_dict.items():
        serializable = {}
        for k, v in data.items():
            if isinstance(v, (np.integer,)):
                serializable[k] = int(v)
            elif isinstance(v, (np.floating,)):
                serializable[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                serializable[k] = bool(v)
            else:
                serializable[k] = v
        result[device] = serializable
    return result


def get_date_and_devices_from_store():
    if DATA_STORE['processed_df'] is not None:
        df = DATA_STORE['processed_df']
        min_date, max_date = get_date_range(df)
        devices = get_device_list(df)
        device_options = [{'label': d, 'value': d} for d in devices]
        return min_date, max_date, device_options, devices
    return None, None, [], []


@app.callback(
    [Output('start-date-picker', 'date'),
     Output('end-date-picker', 'date'),
     Output('device-dropdown', 'options'),
     Output('device-dropdown', 'value')],
    [Input('data-loaded-store', 'data'),
     Input('url', 'pathname'),
     Input('url-device-param', 'data')]
)
def update_overview_dates(data_loaded, pathname, url_device):
    min_date, max_date, device_options, devices = get_date_and_devices_from_store()
    
    device_value = None
    if url_device and url_device in devices:
        device_value = [url_device]
    
    return min_date, max_date, device_options, device_value


@app.callback(
    [Output('report-start-date', 'date'),
     Output('report-end-date', 'date')],
    [Input('data-loaded-store', 'data'),
     Input('url', 'pathname')]
)
def update_report_dates(data_loaded, pathname):
    min_date, max_date, _, _ = get_date_and_devices_from_store()
    return min_date, max_date


@app.callback(
    Output('factor-trend-device', 'options'),
    [Input('data-loaded-store', 'data'),
     Input('url', 'pathname')]
)
def update_factor_trend_devices(data_loaded, pathname):
    _, _, device_options, _ = get_date_and_devices_from_store()
    return device_options


@app.callback(
    Output('shift-compare-device', 'options'),
    [Input('data-loaded-store', 'data'),
     Input('url', 'pathname')]
)
def update_shift_compare_devices(data_loaded, pathname):
    _, _, device_options, _ = get_date_and_devices_from_store()
    return device_options


@app.callback(
    [Output('kpi-cards', 'children'),
     Output('oee-breakdown-chart', 'figure'),
     Output('device-oee-ranking', 'figure'),
     Output('overview-gantt-chart', 'figure')],
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('device-dropdown', 'value')]
)
def update_overview(start_date, end_date, selected_devices):
    if DATA_STORE['processed_df'] is None or not start_date or not end_date:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="请先导入数据")
        return [
            get_kpi_card("整体OEE", "--", "请先导入数据", "primary", "🎯"),
            get_kpi_card("可用率", "--", "请先导入数据", "success", "⚙️"),
            get_kpi_card("性能率", "--", "请先导入数据", "warning", "⚡"),
            get_kpi_card("质量率", "--", "请先导入数据", "info", "✅"),
        ], empty_fig, empty_fig, empty_fig
    
    df = DATA_STORE['processed_df']
    
    overall = calculate_oee_overall(df, start_date, end_date, DEFAULT_TAKT)
    summary = overall['设备汇总']
    
    oee_value = f"{summary['OEE']*100:.2f}%"
    avail_value = f"{summary['可用率']*100:.2f}%"
    perf_value = f"{summary['性能率']*100:.2f}%"
    qual_value = f"{summary['质量率']*100:.2f}%"
    
    oee_color = "success" if summary['OEE'] >= 0.85 else "warning" if summary['OEE'] >= 0.7 else "danger"
    avail_color = "success" if summary['可用率'] >= 0.9 else "warning" if summary['可用率'] >= 0.8 else "danger"
    
    kpi_cards = [
        dbc.Col(get_kpi_card("整体OEE", oee_value, "目标: 85%", oee_color, "🎯"), width=3),
        dbc.Col(get_kpi_card("可用率", avail_value, "目标: 90%", avail_color, "⚙️"), width=3),
        dbc.Col(get_kpi_card("性能率", perf_value, "目标: 95%", "warning", "⚡"), width=3),
        dbc.Col(get_kpi_card("质量率", qual_value, "目标: 99.9%", "info", "✅"), width=3),
    ]
    
    fig_breakdown = go.Figure()
    fig_breakdown.add_trace(go.Bar(
        name='实际值',
        x=['可用率', '性能率', '质量率', 'OEE'],
        y=[summary['可用率']*100, summary['性能率']*100, summary['质量率']*100, summary['OEE']*100],
        marker_color=['#2ecc71', '#f39c12', '#3498db', '#9b59b6'],
        text=[f"{v:.1f}%" for v in [summary['可用率']*100, summary['性能率']*100, summary['质量率']*100, summary['OEE']*100]],
        textposition='outside',
    ))
    fig_breakdown.add_trace(go.Scatter(
        name='目标值',
        x=['可用率', '性能率', '质量率', 'OEE'],
        y=[90, 95, 99.9, 85],
        mode='markers',
        marker=dict(color='#e74c3c', size=10, symbol='line-ns'),
    ))
    fig_breakdown.update_layout(
        title='OEE三因子与目标对比',
        yaxis_title='效率 (%)',
        yaxis=dict(range=[0, 105]),
        barmode='group',
        showlegend=True,
    )
    
    from src.trend_analysis import create_device_comparison_chart
    fig_ranking = create_device_comparison_chart(df, start_date, end_date, DEFAULT_TAKT)
    
    fig_gantt = create_gantt_chart(df, start_date, end_date, selected_devices)
    
    return kpi_cards, fig_breakdown, fig_ranking, fig_gantt


@app.callback(
    Output('health-score-cards', 'children'),
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('device-dropdown', 'value')]
)
def update_health_score_cards(start_date, end_date, selected_devices):
    if DATA_STORE['processed_df'] is None or not end_date:
        return html.Div("请先导入数据", className="text-muted text-center p-3")
    
    df = DATA_STORE['processed_df']
    
    health_scores = calculate_all_health_scores(df, end_date, DEFAULT_TAKT, 7)
    
    if selected_devices:
        health_scores = {k: v for k, v in health_scores.items() if k in selected_devices}
    
    if not health_scores:
        return html.Div("暂无设备数据", className="text-muted text-center p-3")
    
    cards = []
    for device_id, score_data in sorted(health_scores.items()):
        score = score_data['健康分']
        level = get_health_level(score)
        
        card = dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.H6(device_id, className="mb-1", style={"fontWeight": "bold"}),
                            html.Div([
                                html.Span(
                                    level['name'],
                                    style={
                                        'backgroundColor': level['bg_color'],
                                        'color': level['color'],
                                        'padding': '2px 10px',
                                        'borderRadius': '12px',
                                        'fontSize': '0.75rem',
                                        'fontWeight': 'bold',
                                    }
                                ),
                            ]),
                        ]),
                        html.Div(
                            f"{score}",
                            style={
                                'fontSize': '2.5rem',
                                'fontWeight': 'bold',
                                'color': level['color'],
                                'lineHeight': '1',
                            }
                        ),
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                    html.Hr(className="my-2"),
                    html.Div([
                        html.Small(
                            f"可用率: {score_data['可用率']*100:.1f}% | "
                            f"性能率: {score_data['性能率']*100:.1f}% | "
                            f"质量率: {score_data['质量率']*100:.1f}%"
                        ),
                        html.Br(),
                        html.Small(
                            f"稳定性惩罚: -{score_data['稳定性惩罚']:.1f}分",
                            className="text-muted"
                        ),
                    ]),
                ]),
            ], 
            style={'cursor': 'pointer', 'borderLeft': f'4px solid {level["color"]}'},
            className="shadow-sm h-100 health-card",
            id={'type': 'health-card', 'index': device_id}
            ),
            width=3,
            className="mb-3",
        )
        cards.append(card)
    
    return dbc.Row(cards)


@app.callback(
    [Output('url', 'pathname'),
     Output('url-device-param', 'data')],
    [Input({'type': 'health-card', 'index': ALL}, 'n_clicks')],
    [State('url', 'pathname'),
     State('url-device-param', 'data')]
)
def health_card_click(n_clicks, current_path, current_device_param):
    ctx = dash.callback_context
    
    if not n_clicks or all(c is None for c in n_clicks):
        raise dash.exceptions.PreventUpdate
    
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    triggered = ctx.triggered[0]['prop_id']
    if 'health-card' not in triggered:
        raise dash.exceptions.PreventUpdate
    
    trigger_value = ctx.triggered[0]['value']
    if trigger_value is None or trigger_value == 0:
        raise dash.exceptions.PreventUpdate
    
    try:
        import json as _json
        btn_id_str = triggered.split('.')[0]
        btn_id = _json.loads(btn_id_str)
        device_id = btn_id.get('index', '')
    except Exception:
        raise dash.exceptions.PreventUpdate
    
    if device_id:
        return '/downtime-analysis', device_id
    
    raise dash.exceptions.PreventUpdate


@app.callback(
    [Output('drilldown-panel', 'children'),
     Output('drilldown-state', 'data')],
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('device-dropdown', 'value'),
     Input({'type': 'drilldown-btn', 'index': ALL}, 'n_clicks')],
    [State('drilldown-state', 'data')]
)
def update_drilldown(start_date, end_date, selected_devices, btn_clicks, current_state):
    if DATA_STORE['processed_df'] is None or not start_date or not end_date:
        return html.Div("请先导入数据", className="text-muted text-center p-3"), {}
    
    df = DATA_STORE['processed_df']
    overall = calculate_oee_overall(df, start_date, end_date, DEFAULT_TAKT)
    summary = overall['设备汇总']
    devices_data = overall['各设备']
    
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    
    if 'drilldown-btn' in triggered:
        try:
            import json as _json
            btn_id_str = triggered.split('.')[0]
            btn_id = _json.loads(btn_id_str)
            action = btn_id.get('index', '')
        except Exception:
            action = ''
    else:
        action = ''
    
    if not current_state:
        current_state = {}
    
    if action == 'reset' or action == '':
        current_state = {}
    
    if action.startswith('factor:'):
        factor_name = action.split(':')[1]
        current_state = {'level': 'factor', 'factor': factor_name}
    elif action.startswith('device:'):
        parts = action.split(':')
        factor_name = parts[1]
        device_name = parts[2]
        current_state = {'level': 'device', 'factor': factor_name, 'device': device_name}
    elif action.startswith('event:'):
        parts = action.split(':')
        factor_name = parts[1]
        device_name = parts[2]
        date_val = parts[3]
        current_state = {'level': 'event', 'factor': factor_name, 'device': device_name, 'date': date_val}
    
    level = current_state.get('level', 'top')
    
    breadcrumb_items = [html.Span("OEE总览", style={"fontWeight": "bold"})]
    
    if level == 'top':
        oee_val = summary['OEE']
        avail_val = summary['可用率']
        perf_val = summary['性能率']
        qual_val = summary['质量率']
        
        factor_gaps = {
            '可用率': max(0, 0.90 - avail_val),
            '性能率': max(0, 0.95 - perf_val),
            '质量率': max(0, 0.999 - qual_val),
        }
        worst_factor = max(factor_gaps, key=factor_gaps.get)
        
        content = html.Div([
            html.Div([
                html.H5(f"整体OEE: {oee_val*100:.2f}%", className="mb-3"),
                html.P("点击与目标差距最大的因子下钻定位问题根源：", className="text-muted mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("可用率", className="mb-1"),
                            html.H3(f"{avail_val*100:.1f}%", style={"color": "#2ecc71", "fontWeight": "bold"}),
                            html.Small(f"目标 90% | 差距 {factor_gaps['可用率']*100:.1f}个百分点"),
                        ], className="text-center"),
                    ], className="shadow-sm",
                       style={"cursor": "pointer", "border": "2px solid #2ecc71" if worst_factor == '可用率' else "1px solid #dee2e6"}),
                    html.Div(dbc.Button("下钻 →", id={'type': 'drilldown-btn', 'index': 'factor:可用率'},
                                       color="success", size="sm", className="mt-2 w-100"), className="text-center"),
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("性能率", className="mb-1"),
                            html.H3(f"{perf_val*100:.1f}%", style={"color": "#f39c12", "fontWeight": "bold"}),
                            html.Small(f"目标 95% | 差距 {factor_gaps['性能率']*100:.1f}百分点"),
                        ], className="text-center"),
                    ], className="shadow-sm",
                       style={"cursor": "pointer", "border": "2px solid #f39c12" if worst_factor == '性能率' else "1px solid #dee2e6"}),
                    html.Div(dbc.Button("下钻 →", id={'type': 'drilldown-btn', 'index': 'factor:性能率'},
                                       color="warning", size="sm", className="mt-2 w-100"), className="text-center"),
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("质量率", className="mb-1"),
                            html.H3(f"{qual_val*100:.1f}%", style={"color": "#3498db", "fontWeight": "bold"}),
                            html.Small(f"目标 99.9% | 差距 {factor_gaps['质量率']*100:.1f}百分点"),
                        ], className="text-center"),
                    ], className="shadow-sm",
                       style={"cursor": "pointer", "border": "2px solid #3498db" if worst_factor == '质量率' else "1px solid #dee2e6"}),
                    html.Div(dbc.Button("下钻 →", id={'type': 'drilldown-btn', 'index': 'factor:质量率'},
                                       color="info", size="sm", className="mt-2 w-100"), className="text-center"),
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("最大差距因子", className="mb-1"),
                            html.H3(worst_factor, style={"color": "#e74c3c", "fontWeight": "bold"}),
                            html.Small(f"差距 {factor_gaps[worst_factor]*100:.1f}百分点"),
                        ], className="text-center"),
                    ], className="shadow-sm", style={"border": "2px solid #e74c3c"}),
                ], width=3),
            ]),
        ])
        breadcrumb_items = [html.Span("OEE总览", style={"fontWeight": "bold"})]
    
    elif level == 'factor':
        factor_name = current_state['factor']
        factor_col = factor_name
        target_map = {'可用率': 0.90, '性能率': 0.95, '质量率': 0.999}
        target_val = target_map.get(factor_name, 0.85)
        
        device_ranking = []
        for device_name, d in devices_data.items():
            gap = target_val - d[factor_col]
            device_ranking.append({
                'device': device_name,
                'value': d[factor_col],
                'gap': gap,
                'oee': d['OEE'],
            })
        device_ranking.sort(key=lambda x: x['gap'], reverse=True)
        
        rows = []
        for i, dr in enumerate(device_ranking):
            color = "danger" if dr['gap'] > 0.1 else "warning" if dr['gap'] > 0.05 else "success"
            rows.append(html.Tr([
                html.Td(str(i + 1), className="text-center"),
                html.Td(html.Strong(dr['device'])),
                html.Td(f"{dr['value']*100:.2f}%"),
                html.Td(f"{dr['gap']*100:.1f}pp", style={"color": "#dc3545" if dr['gap'] > 0 else "#198754"}),
                html.Td(f"{dr['oee']*100:.2f}%"),
                html.Td(dbc.Button("下钻", id={'type': 'drilldown-btn', 'index': f'device:{factor_name}:{dr["device"]}'},
                                   color=color, size="sm")),
            ]))
        
        content = html.Div([
            html.H5(f"因子: {factor_name}", className="mb-2"),
            html.P(f"整体{factor_name}: {summary[factor_col]*100:.2f}% (目标 {target_val*100:.1f}%)", className="text-muted mb-3"),
            html.P("各设备在该因子的表现排名（差距最大的排在前面）：", className="mb-2"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("#"), html.Th("设备"), html.Th(factor_name), html.Th("与目标差距"), html.Th("OEE"), html.Th("操作"),
                ])),
                html.Tbody(rows),
            ], className="table table-sm"),
        ])
        breadcrumb_items = [
            html.Span("OEE总览", style={"cursor": "pointer", "color": "#0d6efd"}),
            html.Span(" > "),
            html.Span(factor_name, style={"fontWeight": "bold"}),
        ]
    
    elif level == 'device':
        factor_name = current_state['factor']
        device_name = current_state['device']
        factor_col = factor_name
        
        device_df = df[df['设备编号'] == device_name].copy()
        device_df = device_df[(device_df['日期'] >= start_date) & (device_df['日期'] <= end_date)]
        
        health_score_data = calculate_health_score_for_device(
            df, device_name, end_date, DEFAULT_TAKT.get(device_name), 7
        )
        health_score = health_score_data['健康分']
        health_level = get_health_level(health_score)
        
        daily_results = []
        for date_val in sorted(device_df['日期'].unique()):
            day_result = calculate_oee_for_device(df, device_name, date_val, date_val, DEFAULT_TAKT.get(device_name))
            daily_results.append({
                'date': date_val,
                'value': day_result[factor_col],
                'oee': day_result['OEE'],
                'avail': day_result['可用率'],
                'perf': day_result['性能率'],
                'qual': day_result['质量率'],
            })
        daily_results.sort(key=lambda x: x['value'])
        
        rows = []
        for i, dr in enumerate(daily_results):
            color = "danger" if dr['value'] < 0.7 else "warning" if dr['value'] < 0.85 else "success"
            rows.append(html.Tr([
                html.Td(str(i + 1), className="text-center"),
                html.Td(dr['date']),
                html.Td(f"{dr['value']*100:.2f}%"),
                html.Td(f"{dr['oee']*100:.2f}%"),
                html.Td(f"{dr['avail']*100:.1f}%"),
                html.Td(f"{dr['perf']*100:.1f}%"),
                html.Td(f"{dr['qual']*100:.1f}%"),
                html.Td(dbc.Button("下钻", id={'type': 'drilldown-btn', 'index': f'event:{factor_name}:{device_name}:{dr["date"]}'},
                                   color=color, size="sm")),
            ]))
        
        health_card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.H6("💚 设备健康评分", className="mb-1"),
                        html.Small("基于最近7天数据计算", className="text-muted"),
                    ]),
                    html.Div([
                        html.Span(
                            f"{health_score}",
                            style={
                                'fontSize': '2rem',
                                'fontWeight': 'bold',
                                'color': health_level['color'],
                            }
                        ),
                        html.Span(
                            health_level['name'],
                            style={
                                'backgroundColor': health_level['bg_color'],
                                'color': health_level['color'],
                                'padding': '4px 12px',
                                'borderRadius': '15px',
                                'fontSize': '0.875rem',
                                'fontWeight': 'bold',
                                'marginLeft': '12px',
                            }
                        ),
                    ]),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                html.Hr(className="my-2"),
                dbc.Row([
                    dbc.Col([
                        html.Small("可用率"),
                        html.Div(f"{health_score_data['可用率']*100:.1f}%", 
                                 style={'fontWeight': 'bold', 'color': '#2ecc71'}),
                    ], width=3),
                    dbc.Col([
                        html.Small("性能率"),
                        html.Div(f"{health_score_data['性能率']*100:.1f}%", 
                                 style={'fontWeight': 'bold', 'color': '#f39c12'}),
                    ], width=3),
                    dbc.Col([
                        html.Small("质量率"),
                        html.Div(f"{health_score_data['质量率']*100:.1f}%", 
                                 style={'fontWeight': 'bold', 'color': '#3498db'}),
                    ], width=3),
                    dbc.Col([
                        html.Small("稳定性惩罚"),
                        html.Div(f"-{health_score_data['稳定性惩罚']:.1f}分", 
                                 style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                    ], width=3),
                ]),
                html.Small(
                    f"数据天数: {health_score_data['数据天数']}天" + 
                    ("" if health_score_data['数据充足'] else " (数据不足7天，仅供参考)"),
                    className="text-muted mt-2 d-block"
                ),
            ]),
        ], className="shadow-sm mb-3", style={'borderLeft': f'4px solid {health_level["color"]}'})
        
        content = html.Div([
            health_card,
            html.H5(f"设备: {device_name} — {factor_name}逐日明细", className="mb-2 mt-3"),
            html.P("按该因子值从低到高排列，点击下钻查看当日具体停机事件：", className="text-muted mb-3"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("#"), html.Th("日期"), html.Th(factor_name), html.Th("OEE"),
                    html.Th("可用率"), html.Th("性能率"), html.Th("质量率"), html.Th("操作"),
                ])),
                html.Tbody(rows),
            ], className="table table-sm"),
        ])
        breadcrumb_items = [
            html.Span("OEE总览", style={"cursor": "pointer", "color": "#0d6efd"}),
            html.Span(" > "),
            html.Span(factor_name, style={"cursor": "pointer", "color": "#0d6efd"}),
            html.Span(" > "),
            html.Span(device_name, style={"fontWeight": "bold"}),
        ]
    
    elif level == 'event':
        factor_name = current_state['factor']
        device_name = current_state['device']
        date_val = current_state['date']
        factor_col = factor_name
        
        device_df = df[(df['设备编号'] == device_name) & (df['日期'] == date_val)].copy()
        
        day_result = calculate_oee_for_device(df, device_name, date_val, date_val, DEFAULT_TAKT.get(device_name))
        
        non_run = device_df[device_df['记录类型'] != '运行'].sort_values('开始时间戳')
        
        event_rows = []
        for i, (_, row) in enumerate(non_run.iterrows()):
            rtype = row['记录类型']
            reason = row.get('停机原因分类', '')
            duration = row['持续时间分钟']
            start_t = row['开始时间戳'].strftime('%H:%M:%S')
            end_t = row['结束时间戳'].strftime('%H:%M:%S')
            
            if rtype == '停机':
                color = "danger"
                label = f"停机: {reason}" if reason else "停机: 未分类"
            elif rtype == '换模':
                color = "warning"
                label = "换模"
            elif rtype == '空转':
                color = "info"
                label = "空转"
            else:
                color = "secondary"
                label = rtype
            
            event_rows.append(html.Tr([
                html.Td(str(i + 1), className="text-center"),
                html.Td(label),
                html.Td(f"{start_t} - {end_t}"),
                html.Td(f"{duration:.1f} min"),
                html.Td(dbc.Badge(rtype, color=color)),
            ]))
        
        run_df = device_df[device_df['记录类型'] == '运行']
        total_output = run_df['产量'].sum()
        total_good = run_df['合格品数'].sum()
        
        content = html.Div([
            html.H5(f"{device_name} — {date_val} 详细事件", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Small(f"{factor_name}"),
                            html.H4(f"{day_result[factor_col]*100:.2f}%", style={"fontWeight": "bold", "color": "#e74c3c" if day_result[factor_col] < 0.8 else "#198754"}),
                        ], className="text-center p-2"),
                    ], className="shadow-sm"),
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Small("OEE"),
                            html.H4(f"{day_result['OEE']*100:.2f}%", style={"fontWeight": "bold"}),
                        ], className="text-center p-2"),
                    ], className="shadow-sm"),
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Small("产量/合格品"),
                            html.H4(f"{total_output:.0f}/{total_good:.0f}", style={"fontWeight": "bold"}),
                        ], className="text-center p-2"),
                    ], className="shadow-sm"),
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Small("停机次数"),
                            html.H4(f"{len(non_run)}次", style={"fontWeight": "bold", "color": "#e74c3c" if len(non_run) > 3 else "#198754"}),
                        ], className="text-center p-2"),
                    ], className="shadow-sm"),
                ], width=3),
            ], className="mb-3"),
            html.P("当日非运行事件明细：", className="mb-2"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("#"), html.Th("类型/原因"), html.Th("时间段"), html.Th("持续时间"), html.Th("分类"),
                ])),
                html.Tbody(event_rows if event_rows else html.Tr(html.Td("无停机事件", colSpan=5, className="text-center text-muted"))),
            ], className="table table-sm"),
        ])
        breadcrumb_items = [
            html.Span("OEE总览", style={"cursor": "pointer", "color": "#0d6efd"}),
            html.Span(" > "),
            html.Span(factor_name, style={"cursor": "pointer", "color": "#0d6efd"}),
            html.Span(" > "),
            html.Span(device_name, style={"cursor": "pointer", "color": "#0d6efd"}),
            html.Span(" > "),
            html.Span(f"{date_val}", style={"fontWeight": "bold"}),
        ]
    
    breadcrumb = html.Div([
        dbc.Button("⟲ 重置", id={'type': 'drilldown-btn', 'index': 'reset'},
                   color="secondary", size="sm", className="me-2"),
        *breadcrumb_items,
    ], className="mb-3")
    
    return html.Div([breadcrumb, content]), current_state


@app.callback(
    [Output('oee-trend-content', 'style'),
     Output('health-trend-content', 'style')],
    [Input('trend-tabs', 'active_tab')]
)
def toggle_trend_tabs(active_tab):
    if active_tab == 'oee-trend':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


@app.callback(
    Output('health-trend-devices', 'options'),
    [Input('data-loaded-store', 'data'),
     Input('url', 'pathname')]
)
def update_health_trend_devices(data_loaded, pathname):
    _, _, device_options, _ = get_date_and_devices_from_store()
    return device_options


@app.callback(
    Output('health-trend-devices', 'value'),
    [Input('health-trend-devices', 'options')]
)
def set_default_health_devices(options):
    if options and len(options) > 0:
        return [options[0]['value']]
    return []


@app.callback(
    Output('health-trend-chart', 'figure'),
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('health-trend-devices', 'value')]
)
def update_health_trend_chart(start_date, end_date, selected_devices):
    if DATA_STORE['processed_df'] is None or not start_date or not end_date:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="请先导入数据")
        return empty_fig
    
    df = DATA_STORE['processed_df']
    
    if not selected_devices:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="请选择至少一台设备")
        return empty_fig
    
    if len(selected_devices) > 5:
        selected_devices = selected_devices[:5]
    
    fig = go.Figure()
    
    import plotly.express as px
    
    colors = px.colors.qualitative.Set1
    
    for idx, device_id in enumerate(selected_devices):
        trend_data = calculate_health_score_trend(
            df, device_id, start_date, end_date, 
            DEFAULT_TAKT.get(device_id), 7
        )
        
        if len(trend_data) == 0:
            continue
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=trend_data['日期'],
                y=trend_data['健康分'],
                mode='lines+markers',
                name=f'{device_id} 健康分',
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )
    
    fig.add_hline(
        y=90,
        line_dash="dash",
        line_color="#198754",
        annotation_text="优秀 (90分)",
        annotation_position="top right",
        opacity=0.7,
    )
    
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#0d6efd",
        annotation_text="良好 (70分)",
        annotation_position="top right",
        opacity=0.7,
    )
    
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="#fd7e14",
        annotation_text="关注 (50分)",
        annotation_position="top right",
        opacity=0.7,
    )
    
    fig.update_layout(
        title='设备健康分趋势图（7日滑动窗口）',
        xaxis_title='日期',
        yaxis_title='健康分',
        yaxis=dict(range=[0, 105]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    
    return fig


@app.callback(
    [Output('trend-chart', 'figure'),
     Output('factor-trend-chart', 'figure')],
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('device-dropdown', 'value'),
     Input('trend-type', 'value'),
     Input('trend-options', 'value'),
     Input('factor-trend-device', 'value')]
)
def update_trend_charts(start_date, end_date, selected_devices, trend_type, options, factor_device):
    if DATA_STORE['processed_df'] is None or not start_date or not end_date:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="请先导入数据")
        return empty_fig, empty_fig
    
    df = DATA_STORE['processed_df']
    devices = selected_devices if selected_devices else []
    
    show_ma = 'moving_avg' in options
    
    fig_trend = create_trend_chart(
        df, devices, start_date, end_date, trend_type, DEFAULT_TAKT, 0.85, show_ma
    )
    
    if factor_device:
        fig_factor = create_factor_trend_chart(
            df, factor_device, start_date, end_date, 
            DEFAULT_TAKT.get(factor_device), trend_type
        )
    else:
        fig_factor = go.Figure()
        fig_factor.update_layout(title="请选择设备")
    
    return fig_trend, fig_factor


@app.callback(
    Output('downtime-tab-content', 'children'),
    [Input('downtime-tabs', 'active_tab'),
     Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('device-dropdown', 'value')]
)
def render_downtime_tab(active_tab, start_date, end_date, selected_devices):
    if DATA_STORE['processed_df'] is None:
        return html.Div("请先导入数据", className="text-muted p-5 text-center")
    
    df = DATA_STORE['processed_df']
    devices = selected_devices if selected_devices else None
    
    if active_tab == 'pareto':
        fig = create_pareto_chart(df, start_date, end_date, devices)
        return dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig),
            ]),
        ], className="shadow-sm")
    
    elif active_tab == 'gantt':
        fig = create_gantt_chart(df, start_date, end_date, devices)
        return dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig),
            ]),
        ], className="shadow-sm")
    
    elif active_tab == 'pie':
        return dbc.Card([
            dbc.CardHeader([
                dbc.RadioItems(
                    id='pie-dimension',
                    options=[
                        {'label': '按设备', 'value': '设备'},
                        {'label': '按班次', 'value': '班次'},
                        {'label': '按日期', 'value': '日期'},
                    ],
                    value='设备',
                    inline=True,
                ),
            ]),
            dbc.CardBody([
                dcc.Graph(id='downtime-pie-chart'),
            ]),
        ], className="shadow-sm")
    
    elif active_tab == 'drilldown':
        drilldown_data = create_root_cause_drilldown(df, start_date, end_date, devices)
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("OEE因子分解"),
                        dbc.CardBody([
                            html.H4(f"整体OEE: {drilldown_data['overall_oee']*100:.2f}%", className="text-primary"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(f"可用率: {drilldown_data['overall_availability']*100:.2f}%"),
                                    dbc.Progress(value=drilldown_data['overall_availability']*100, color="success"),
                                ]),
                                dbc.Col([
                                    html.Div(f"性能率: {drilldown_data['overall_performance']*100:.2f}%"),
                                    dbc.Progress(value=drilldown_data['overall_performance']*100, color="warning"),
                                ]),
                                dbc.Col([
                                    html.Div(f"质量率: {drilldown_data['overall_quality']*100:.2f}%"),
                                    dbc.Progress(value=drilldown_data['overall_quality']*100, color="info"),
                                ]),
                            ]),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("各设备明细"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                data=drilldown_data['device_details'],
                                columns=[
                                    {"name": "设备", "id": "设备"},
                                    {"name": "OEE", "id": "OEE", "format": {"specifier": ".1%"}},
                                    {"name": "可用率", "id": "可用率", "format": {"specifier": ".1%"}},
                                    {"name": "性能率", "id": "性能率", "format": {"specifier": ".1%"}},
                                    {"name": "质量率", "id": "质量率", "format": {"specifier": ".1%"}},
                                ],
                                page_size=10,
                                sort_action='native',
                                style_table={'overflowX': 'auto'},
                            ),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("停机原因排名 (Pareto)"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                data=drilldown_data['pareto_data'],
                                columns=[
                                    {"name": "停机原因", "id": "停机原因"},
                                    {"name": "持续时间(分钟)", "id": "持续时间分钟", "type": "numeric"},
                                    {"name": "次数", "id": "次数", "type": "numeric"},
                                ],
                                page_size=10,
                                sort_action='native',
                                style_table={'overflowX': 'auto'},
                            ),
                        ]),
                    ], className="shadow-sm"),
                ], width=12),
            ]),
        ])
    
    return html.Div("")


@app.callback(
    Output('downtime-pie-chart', 'figure'),
    [Input('pie-dimension', 'value'),
     Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('device-dropdown', 'value')]
)
def update_pie_chart(dimension, start_date, end_date, selected_devices):
    if DATA_STORE['processed_df'] is None:
        return go.Figure()
    
    df = DATA_STORE['processed_df']
    devices = selected_devices if selected_devices else None
    
    return create_downtime_pie(df, start_date, end_date, dimension, devices)


@app.callback(
    [Output('benchmark-chart', 'figure'),
     Output('benchmark-device-ranking', 'figure'),
     Output('shift-comparison-chart', 'figure')],
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('device-dropdown', 'value'),
     Input('shift-compare-device', 'value')]
)
def update_benchmark(start_date, end_date, selected_devices, shift_device):
    if DATA_STORE['processed_df'] is None or not start_date or not end_date:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="请先导入数据")
        return empty_fig, empty_fig, empty_fig
    
    df = DATA_STORE['processed_df']
    
    overall = calculate_oee_overall(df, start_date, end_date, DEFAULT_TAKT)
    summary = overall['设备汇总']
    
    fig_benchmark = create_benchmark_chart(
        summary['OEE'], summary['可用率'], summary['性能率'], summary['质量率']
    )
    
    from src.trend_analysis import create_device_comparison_chart
    fig_ranking = create_device_comparison_chart(df, start_date, end_date, DEFAULT_TAKT)
    
    if shift_device:
        fig_shift = create_shift_comparison_chart(
            df, shift_device, start_date, end_date,
            DEFAULT_TAKT.get(shift_device)
        )
    else:
        fig_shift = go.Figure()
        fig_shift.update_layout(title="请选择设备")
    
    return fig_benchmark, fig_ranking, fig_shift


@app.callback(
    Output('anomaly-cards', 'children'),
    [Input('anomaly-oee-target', 'value'),
     Input('anomaly-drop-threshold', 'value'),
     Input('anomaly-consecutive-days', 'value'),
     Input('previous-health-scores', 'data'),
     Input('current-health-scores', 'data')]
)
def update_anomaly_detection(oee_target, drop_threshold, consecutive_days, prev_scores, current_scores):
    if DATA_STORE['processed_df'] is None:
        return html.Div("请先导入数据", className="text-muted p-5 text-center")
    
    df = DATA_STORE['processed_df']
    
    anomalies = []
    devices = df['设备编号'].unique()
    
    from src.trend_analysis import detect_anomalies
    
    for device in devices:
        device_anomalies = detect_anomalies(
            df, device, DEFAULT_TAKT.get(device),
            threshold_drop=drop_threshold/100,
            oee_target=oee_target,
            consecutive_days=consecutive_days,
        )
        anomalies.extend(device_anomalies)
    
    if prev_scores and current_scores:
        health_drop_anomalies = detect_health_score_drop_anomalies(
            df, current_scores, prev_scores, drop_threshold=15
        )
        anomalies.extend(health_drop_anomalies)
    
    if not anomalies:
        return dbc.Alert("✅ 未检测到异常，所有设备运行正常！", color="success", className="text-center")
    
    cards = []
    for idx, a in enumerate(anomalies):
        if a['类型'] == 'OEE骤降告警':
            color = "danger"
            icon = "📉"
            title = f"OEE骤降告警 - {a['设备']}"
            body = html.Div([
                html.H6(f"日期: {a['日期']}"),
                html.P(f"当前OEE: {a['当前OEE']*100:.2f}%"),
                html.P(f"7日均值: {a['7日均值']*100:.2f}%"),
                html.P(f"下降幅度: {a['下降幅度']*100:.1f} 个百分点"),
                html.P(f"主要归因: {a['主要归因因子']} (下降{a['因子下降']*100:.1f}个百分点)"),
            ])
        elif a['类型'] == '健康分骤降':
            color = "danger"
            icon = "💔"
            title = f"健康分骤降告警 - {a['设备']}"
            body = html.Div([
                html.H6("健康评分对比"),
                html.P([
                    html.Strong("上次评分: "),
                    html.Span(f"{a['上次评分']}分", style={'color': '#198754', 'fontWeight': 'bold'}),
                    html.Span(" → "),
                    html.Strong("本次评分: "),
                    html.Span(f"{a['本次评分']}分", style={'color': '#dc3545', 'fontWeight': 'bold'}),
                ]),
                html.P(f"下降幅度: {a['下降幅度']} 分", 
                       style={'color': '#dc3545', 'fontWeight': 'bold'}),
                html.Hr(className="my-2"),
                html.H6("初步归因分析"),
                html.P(f"最大降幅因子: {a['主要归因因子']}"),
                html.P(f"该因子下降: {a['因子下降']*100:.1f} 个百分点"),
                html.Small("点击跳转到停机归因分析页面查看详情", className="text-muted"),
            ])
        else:
            color = "warning"
            icon = "⚠️"
            title = f"持续低效预警 - {a['设备']}"
            body = html.Div([
                html.H6(f"日期范围: {a['开始日期']} 至 {a['结束日期']}"),
                html.P(f"持续天数: {a['持续天数']} 天"),
                html.P(f"平均OEE: {a['平均OEE']*100:.2f}%"),
                html.P(f"目标值: {a['目标值']*100:.0f}%"),
            ])
        
        cards.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader([
                        html.Span(icon, className="me-2"),
                        title,
                    ]),
                    dbc.CardBody(body),
                ], color=color, outline=True, className="shadow-sm h-100"),
                width=4,
                className="mb-3",
            )
        )
    
    return dbc.Row(cards)


@app.callback(
    [Output('report-download-area', 'children'),
     Output('report-preview', 'children')],
    [Input('generate-report-btn', 'n_clicks')],
    [State('report-type', 'value'),
     State('report-start-date', 'date'),
     State('report-end-date', 'date')]
)
def generate_report(n_clicks, report_type, start_date, end_date):
    if not n_clicks or DATA_STORE['processed_df'] is None:
        return html.Div("请先导入数据并设置报告参数", className="text-muted"), \
               html.Div("报告预览将在此显示", className="text-muted text-center p-5")
    
    try:
        df = DATA_STORE['processed_df']
        pdf_bytes = generate_pdf_report(
            df, start_date, end_date, report_type, DEFAULT_TAKT, 0.85
        )
        
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        download_link = html.A(
            "📥 下载PDF报告",
            href=f"data:application/pdf;base64,{pdf_b64}",
            download=f"OEE分析报告_{start_date}_{end_date}.pdf",
            className="btn btn-success",
        )
        
        preview = html.Div([
            html.H5("报告生成成功！", className="text-success mb-3"),
            html.P(f"报告类型: {report_type}"),
            html.P(f"报告期间: {start_date} 至 {end_date}"),
            html.P("点击上方按钮下载完整PDF报告"),
            html.Hr(),
            html.H6("报告摘要:"),
        ])
        
        return download_link, preview
    
    except Exception as e:
        return dbc.Alert(f"报告生成失败: {str(e)}", color="danger"), \
               html.Div("")


SCHEDULE_STORE = {
    'schedule': None,
}


def create_maintenance_gantt_chart(schedule_result):
    if not schedule_result or not schedule_result.get('排程结果'):
        fig = go.Figure()
        fig.update_layout(title="请先导入数据并生成排程")
        return fig

    assignments = schedule_result['排程结果']
    device_infos = schedule_result['设备信息']
    current_time = schedule_result.get('当前时间', datetime.now())
    if hasattr(current_time, 'to_pydatetime'):
        current_time = current_time.to_pydatetime()
    horizon_days = 7

    device_list = sorted([info['设备编号'] for info in device_infos])

    fig = go.Figure()

    for device_idx, device_id in enumerate(device_list):
        device_info = next((i for i in device_infos if i['设备编号'] == device_id), None)
        device_assignments = [a for a in assignments if a['设备编号'] == device_id]

        for assignment in device_assignments:
            start = assignment['维护开始时间']
            end = assignment['维护结束时间']
            urgency = assignment['紧迫度']
            color = get_urgency_color(urgency)

            hover_text = (
                f"设备: {device_id}<br>"
                f"紧迫度: {urgency}<br>"
                f"维护时段: {start.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%H:%M')}<br>"
                f"班次: {assignment['班次']} ({assignment['时段类型']})<br>"
                f"健康分: {assignment['健康分']}分<br>"
            )
            if assignment.get('当前可靠度') is not None:
                hover_text += f"当前可靠度: {assignment['当前可靠度']*100:.1f}%<br>"
            if assignment.get('建议窗口Δt小时') is not None:
                hover_text += f"建议维护窗口: {assignment['建议窗口Δt小时']:.1f}小时后<br>"
            hover_text += f"分配方式: {assignment['分配备注']}"

            if assignment.get('强制优先'):
                hover_text += "<br><b>⚠️ 健康分低于阈值，强制优先</b>"

            fig.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[device_idx, device_idx],
                    mode='lines',
                    line=dict(color=color, width=25),
                    name=f"{device_id} - {urgency}",
                    showlegend=False,
                    hovertext=hover_text,
                    hoverinfo='text',
                )
            )

            if assignment.get('建议维护时间'):
                pass

    fig.add_shape(
        type="line",
        x0=current_time,
        x1=current_time,
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color="#0d6efd", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=current_time,
        y=1.0,
        yref='paper',
        text="当前时间",
        showarrow=False,
        xanchor='left',
        font=dict(color="#0d6efd"),
    )

    fig.update_yaxes(
        tickvals=list(range(len(device_list))),
        ticktext=device_list,
        autorange='reversed',
        title='设备',
    )

    horizon_end = current_time + timedelta(days=horizon_days)
    fig.update_xaxes(
        title='时间',
        range=[current_time - timedelta(hours=2), horizon_end + timedelta(hours=2)],
    )

    legend_items = [
        ('紧急', '#dc3545'),
        ('临近', '#fd7e14'),
        ('充裕', '#198754'),
        ('数据不足', '#6c757d'),
    ]

    for name, color in legend_items:
        count = sum(1 for a in assignments if a['紧迫度'] == name)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=color, width=10),
                name=f"{name} ({count}台)",
                showlegend=True,
            )
        )

    fig.update_layout(
        title='预防性维护排程甘特图（未来7天）',
        height=max(400, len(device_list) * 60),
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_schedule_list_table(schedule_result):
    if not schedule_result or not schedule_result.get('排程结果'):
        return html.Div("请先导入数据并生成排程", className="text-muted text-center p-5")

    assignments = sorted(schedule_result['排程结果'], key=lambda x: x['维护开始时间'])
    unassigned = schedule_result.get('未分配设备', [])

    if not assignments:
        content = html.Div("⚠️ 暂无排程结果", className="text-warning text-center p-3")
    else:
        rows = []
        for idx, a in enumerate(assignments, 1):
            urgency = a['紧迫度']
            color = get_urgency_color(urgency)
            badge_color = {
                '紧急': 'danger',
                '临近': 'warning',
                '充裕': 'success',
                '数据不足': 'secondary',
            }.get(urgency, 'secondary')

            rows.append(html.Tr([
                html.Td(str(idx), className="text-center"),
                html.Td(html.Strong(a['设备编号'])),
                html.Td([
                    html.Div(f"{a['维护开始时间'].strftime('%Y-%m-%d')}"),
                    html.Small(
                        f"{a['维护开始时间'].strftime('%H:%M')} - {a['维护结束时间'].strftime('%H:%M')}",
                        className="text-muted"
                    ),
                ]),
                html.Td(f"{a['班次']}班"),
                html.Td(a['时段类型']),
                html.Td(f"{a['维护时长分钟']}分钟"),
                html.Td(dbc.Badge(urgency, color=badge_color)),
                html.Td([
                    html.Span(
                        f"{a['健康分']}分",
                        style={
                            'fontWeight': 'bold',
                            'color': '#dc3545' if a['健康分'] < 50 else '#fd7e14' if a['健康分'] < 70 else '#198754'
                        }
                    ),
                ]),
                html.Td(
                    f"{a['当前可靠度']*100:.1f}%" if a.get('当前可靠度') is not None else
                    html.Span("N/A", className="text-muted")
                ),
                html.Td(
                    f"{a['建议窗口Δt小时']:.1f}h" if a.get('建议窗口Δt小时') is not None else
                    html.Span("N/A", className="text-muted")
                ),
                html.Td([
                    html.Span(a['分配备注']),
                    html.Br(),
                    html.Small(
                        "数据不足(固定周期)" if a.get('使用默认周期') else
                        ("威布尔拟合" if a.get('数据充足') else "数据不足"),
                        className="text-muted"
                    ),
                ]),
            ]))

        content = html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("#", className="text-center"),
                    html.Th("设备编号"),
                    html.Th("维护时间"),
                    html.Th("班次"),
                    html.Th("时段"),
                    html.Th("时长"),
                    html.Th("紧迫度"),
                    html.Th("健康分"),
                    html.Th("可靠度"),
                    html.Th("建议窗口"),
                    html.Th("分配说明"),
                ])),
                html.Tbody(rows),
            ], className="table table-sm table-hover"),
        ])

    unassigned_content = html.Div("")
    if unassigned:
        unassigned_rows = []
        for idx, u in enumerate(unassigned, 1):
            urgency = u.get('紧迫度', '未知')
            badge_color = {
                '紧急': 'danger',
                '临近': 'warning',
                '充裕': 'success',
                '数据不足': 'secondary',
            }.get(urgency, 'secondary')

            unassigned_rows.append(html.Tr([
                html.Td(str(idx), className="text-center"),
                html.Td(html.Strong(u['设备编号'])),
                html.Td(u['原因']),
                html.Td(dbc.Badge(urgency, color=badge_color)),
                html.Td(
                    f"{u['建议窗口Δt小时']:.1f}小时后" if u.get('建议窗口Δt小时') is not None else "N/A"
                ),
            ]))

        unassigned_content = dbc.Alert([
            html.H6(f"⚠️ 有 {len(unassigned)} 台设备未能在未来7天内分配维护时段：", className="alert-heading"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("#"),
                    html.Th("设备编号"),
                    html.Th("原因"),
                    html.Th("紧迫度"),
                    html.Th("建议维护窗口"),
                ])),
                html.Tbody(unassigned_rows),
            ], className="table table-sm table-warning"),
            html.Hr(),
            html.Small("建议：考虑扩展维护时段或在生产间隙安排额外维护", className="text-muted"),
        ], color="warning", className="mt-4")

    return html.Div([content, unassigned_content])


def create_weibull_params_panel(schedule_result):
    if not schedule_result or not schedule_result.get('设备信息'):
        return html.Div("请先导入数据并生成排程", className="text-muted text-center p-5")

    def _clean(children_list):
        return [c for c in children_list if c is not None]

    device_infos = schedule_result['设备信息']
    stats = schedule_result.get('统计信息', {})

    cards = []
    for info in sorted(device_infos, key=lambda x: x['设备编号']):
        device_id = info['设备编号']
        health_score = info['健康分']
        health_level = get_health_level(health_score)

        beta = info.get('beta')
        eta = info.get('eta')
        current_R = info.get('当前可靠度')
        delta_t = info.get('建议窗口Δt小时')
        t_since = info.get('距上次故障小时')
        urgency = info.get('紧迫度', '未知')
        data_sufficient = info.get('数据充足', False)
        use_default = info.get('使用默认周期', False)

        urgency_color = get_urgency_color(urgency)

        if use_default:
            beta_display = html.Span("N/A", className="text-muted")
            eta_display = html.Span("N/A", className="text-muted")
            reliability_display = html.Span("N/A (固定周期)", className="text-muted")
            window_display = html.Div([
                html.Span("168小时", style={'fontWeight': 'bold', 'color': '#6c757d'}),
                html.Small(" (默认7天固定周期)", className="text-muted d-block"),
            ])
        else:
            beta_display = html.Span(f"{beta:.3f}" if beta else "N/A", style={'fontWeight': 'bold'})
            eta_display = html.Span(f"{eta:.1f}h" if eta else "N/A", style={'fontWeight': 'bold'})
            reliability_display = html.Span(
                f"{current_R*100:.1f}%" if current_R is not None else "N/A",
                style={
                    'fontWeight': 'bold',
                    'color': '#dc3545' if current_R is not None and current_R < 0.7 else '#198754'
                }
            )
            window_display = html.Div([
                html.Span(
                    f"{delta_t:.1f}小时后" if delta_t is not None else "N/A",
                    style={'fontWeight': 'bold', 'color': urgency_color}
                ),
                html.Small(
                    f" (距上次故障已运行 {t_since:.1f}h)" if t_since is not None else "",
                    className="text-muted d-block"
                ),
            ])

        weibull_interpretation = ""
        if beta is not None and not use_default:
            if beta < 1:
                weibull_interpretation = "早期故障期（磨合期）"
            elif abs(beta - 1) < 0.1:
                weibull_interpretation = "偶发故障期（随机失效）"
            elif beta < 3:
                weibull_interpretation = "损耗故障早期"
            else:
                weibull_interpretation = "损耗故障期（老化）"

        card_body_children = [
            html.Div([
                html.Div([
                    html.H6(device_id, className="mb-1", style={"fontWeight": "bold"}),
                    html.Div([
                        dbc.Badge(
                            urgency,
                            color={
                                '紧急': 'danger',
                                '临近': 'warning',
                                '充裕': 'success',
                                '数据不足': 'secondary',
                            }.get(urgency, 'secondary'),
                            className="me-2"
                        ),
                        html.Span(
                            health_level['name'],
                            style={
                                'backgroundColor': health_level['bg_color'],
                                'color': health_level['color'],
                                'padding': '2px 10px',
                                'borderRadius': '12px',
                                'fontSize': '0.75rem',
                                'fontWeight': 'bold',
                            }
                        ),
                    ], className="mt-1"),
                ]),
                html.Div(
                    f"{health_score}",
                    style={
                        'fontSize': '2rem',
                        'fontWeight': 'bold',
                        'color': health_level['color'],
                        'lineHeight': '1',
                    }
                ),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start'}),
            html.Hr(className="my-2"),
            dbc.Row([
                dbc.Col(_clean([
                    html.Small("形状参数 β", className="text-muted d-block"),
                    beta_display,
                    html.Small(weibull_interpretation, className="text-primary d-block mt-1") if weibull_interpretation else None,
                ]), width=6),
                dbc.Col(_clean([
                    html.Small("尺度参数 η", className="text-muted d-block"),
                    eta_display,
                    html.Small(
                        f"特征寿命 (63.2%失效时)",
                        className="text-muted d-block mt-1"
                    ) if eta and not use_default else None,
                ]), width=6),
            ]),
            html.Hr(className="my-2"),
            dbc.Row([
                dbc.Col(_clean([
                    html.Small("当前可靠度 R(t)", className="text-muted d-block"),
                    reliability_display,
                ]), width=6),
                dbc.Col(_clean([
                    html.Small("建议维护窗口 (R≥70%)", className="text-muted d-block"),
                    window_display,
                ]), width=6),
            ]),
            html.Hr(className="my-2"),
            html.Small(
                f"历史故障记录: {info.get('故障次数', 0)}次 | "
                f"故障间隔数据: {info.get('故障间隔数', 0)}条 | "
                f"数据{'充足 ✓' if data_sufficient else '不足 ⚠️ (建议固定周期维护)'}",
                className=
                    "d-block mt-2 " + 
                    ("text-success" if data_sufficient else "text-warning")
            ),
        ]
        if info.get('强制优先', False):
            card_body_children.append(
                dbc.Badge(
                    "⚠️ 健康分<50，强制优先维护",
                    color="danger",
                    className="w-100 mt-2 py-2"
                )
            )
        card_body_children = [c for c in card_body_children if c is not None]

        card = dbc.Col(
            dbc.Card([
                dbc.CardBody(card_body_children),
            ],
            style={
                'borderLeft': f'4px solid {urgency_color}',
            },
            className="shadow-sm h-100"
            ),
            width=4,
            className="mb-3",
        )
        cards.append(card)

    summary_card = dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H6("📈 总体统计", className="mb-3", style={'fontWeight': 'bold'}),
                dbc.Row([
                    dbc.Col([
                        html.Small("总设备数", className="text-muted d-block"),
                        html.H5(f"{stats.get('总设备数', 0)}", style={'fontWeight': 'bold', 'color': '#0d6efd'}),
                    ], width=6),
                    dbc.Col([
                        html.Small("已分配排程", className="text-muted d-block"),
                        html.H5(f"{stats.get('已分配', 0)}", style={'fontWeight': 'bold', 'color': '#198754'}),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Small("数据充足", className="text-muted d-block"),
                        html.H5(f"{stats.get('数据充足', 0)}", style={'fontWeight': 'bold', 'color': '#20c997'}),
                    ], width=6),
                    dbc.Col([
                        html.Small("紧急设备", className="text-muted d-block"),
                        html.H5(f"{stats.get('紧急设备', 0)}", style={'fontWeight': 'bold', 'color': '#dc3545'}),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Small("数据不足", className="text-muted d-block"),
                        html.H5(f"{stats.get('数据不足', 0)}", style={'fontWeight': 'bold', 'color': '#6c757d'}),
                    ], width=6),
                    dbc.Col([
                        html.Small("未分配", className="text-muted d-block"),
                        html.H5(f"{stats.get('未分配', 0)}", style={'fontWeight': 'bold', 'color': '#fd7e14'}),
                    ], width=6),
                ]),
            ]),
        ], className="shadow-sm h-100 bg-light"),
        width=4,
        className="mb-3",
    )

    all_cards = [summary_card] + cards

    return dbc.Row(all_cards)


def generate_ics_file(schedule_result):
    if not schedule_result or not schedule_result.get('排程结果'):
        return None

    assignments = schedule_result['排程结果']
    current_time = schedule_result.get('当前时间', datetime.now())

    urgency_map = {
        '紧急': 'HIGH',
        '临近': 'MEDIUM',
        '充裕': 'LOW',
        '数据不足': 'NORMAL',
    }

    lines = []
    lines.append("BEGIN:VCALENDAR")
    lines.append("VERSION:2.0")
    lines.append("PRODID:-//OEE Dashboard//Maintenance Schedule//CN")
    lines.append("CALSCALE:GREGORIAN")
    lines.append("METHOD:PUBLISH")
    lines.append(f"X-WR-CALNAME:OEE维保排程 - {current_time.strftime('%Y%m%d')}")
    lines.append(f"X-WR-TIMEZONE:Asia/Shanghai")

    uid_counter = 0
    for a in assignments:
        uid_counter += 1
        device_id = a['设备编号']
        start = a['维护开始时间']
        end = a['维护结束时间']
        urgency = a['紧迫度']
        priority = urgency_map.get(urgency, 'NORMAL')

        dtstamp = datetime.now().strftime('%Y%m%dT%H%M%SZ')
        dtstart = start.strftime('%Y%m%dT%H%M%S')
        dtend = end.strftime('%Y%m%dT%H%M%S')

        summary = f"【{urgency}】预防性维护 - {device_id}"

        description_parts = [
            f"设备编号: {device_id}",
            f"维护类型: 预防性维护",
            f"紧迫等级: {urgency}",
            f"维护时长: {a['维护时长分钟']}分钟",
            f"班次: {a['班次']}班 ({a['时段类型']})",
            f"健康分: {a['健康分']}分",
        ]
        if a.get('当前可靠度') is not None:
            description_parts.append(f"当前可靠度: {a['当前可靠度']*100:.1f}%")
        if a.get('建议窗口Δt小时') is not None:
            description_parts.append(f"建议维护窗口: {a['建议窗口Δt小时']:.1f}小时后")
        if a.get('威布尔beta') is not None:
            description_parts.append(f"威布尔参数: β={a['威布尔beta']:.3f}, η={a['威布尔eta']:.1f}h")
        description_parts.append(f"分配方式: {a['分配备注']}")
        if a.get('强制优先'):
            description_parts.append("⚠️ 健康分低于阈值，强制优先维护")

        description = "\\n".join(description_parts)

        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:oee-maint-{uid_counter}-{start.strftime('%Y%m%d%H%M')}@oee-dashboard")
        lines.append(f"DTSTAMP:{dtstamp}")
        lines.append(f"DTSTART:{dtstart}")
        lines.append(f"DTEND:{dtend}")
        lines.append(f"SUMMARY:{summary}")
        lines.append(f"DESCRIPTION:{description}")
        lines.append(f"PRIORITY:{'1' if priority == 'HIGH' else '3' if priority == 'MEDIUM' else '5'}")
        lines.append(f"STATUS:CONFIRMED")
        lines.append("TRANSP:OPAQUE")
        lines.append(f"LOCATION:车间 - {device_id}工位")
        lines.append(f"CATEGORIES:MAINTENANCE,OEE,{urgency.upper()}")
        if priority == 'HIGH':
            lines.append("BEGIN:VALARM")
            lines.append("TRIGGER:-PT24H")
            lines.append("ACTION:DISPLAY")
            lines.append("DESCRIPTION:紧急维护任务提醒")
            lines.append("END:VALARM")
            lines.append("BEGIN:VALARM")
            lines.append("TRIGGER:-PT1H")
            lines.append("ACTION:DISPLAY")
            lines.append("DESCRIPTION:紧急维护任务即将开始")
            lines.append("END:VALARM")
        elif priority == 'MEDIUM':
            lines.append("BEGIN:VALARM")
            lines.append("TRIGGER:-PT12H")
            lines.append("ACTION:DISPLAY")
            lines.append("DESCRIPTION:维护任务提醒")
            lines.append("END:VALARM")
        lines.append("END:VEVENT")

    lines.append("END:VCALENDAR")

    return "\r\n".join(lines)


@app.callback(
    [Output('schedule-kpi-cards', 'children'),
     Output('maintenance-gantt-chart', 'figure'),
     Output('schedule-list', 'children'),
     Output('weibull-params-panel', 'children')],
    [Input('generate-schedule-btn', 'n_clicks'),
     Input('schedule-horizon', 'value'),
     Input('reliability-threshold', 'value'),
     Input('health-threshold', 'value')]
)
def generate_maintenance_schedule(n_clicks, horizon, rel_threshold, health_thresh):
    if DATA_STORE['processed_df'] is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="请先导入数据")
        no_data = html.Div("请先导入数据", className="text-muted text-center p-5")
        kpi_cards = [
            dbc.Col(get_kpi_card("总设备数", "--", "请先导入数据", "primary", "🔧"), width=3),
            dbc.Col(get_kpi_card("已分配", "--", "请先导入数据", "success", "✅"), width=3),
            dbc.Col(get_kpi_card("紧急设备", "--", "请先导入数据", "danger", "⚠️"), width=3),
            dbc.Col(get_kpi_card("数据充足", "--", "请先导入数据", "info", "📊"), width=3),
        ]
        return kpi_cards, empty_fig, no_data, no_data

    if not n_clicks:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="点击上方'生成排程'按钮开始")
        hint = html.Div([
            dbc.Alert([
                html.H5("👆 请设置参数后点击'生成排程'按钮", className="alert-heading"),
                html.P("系统将基于威布尔分布预测设备故障时间，并自动优化维护排程。", className="mb-0"),
            ], color="info", className="text-center"),
        ], className="p-5")
        kpi_cards = [
            dbc.Col(get_kpi_card("总设备数", "--", "生成排程后显示", "primary", "🔧"), width=3),
            dbc.Col(get_kpi_card("已分配", "--", "生成排程后显示", "success", "✅"), width=3),
            dbc.Col(get_kpi_card("紧急设备", "--", "生成排程后显示", "danger", "⚠️"), width=3),
            dbc.Col(get_kpi_card("数据充足", "--", "生成排程后显示", "info", "📊"), width=3),
        ]
        return kpi_cards, empty_fig, hint, hint

    df = DATA_STORE['processed_df']

    horizon_days = max(3, min(30, int(horizon) if horizon else 7))

    from src.predictive_maintenance import (
        DEFAULT_MAINTENANCE_INTERVAL_HOURS,
        RELIABILITY_THRESHOLD as _DEFAULT_THRESH
    )
    import src.predictive_maintenance as pm_module
    if rel_threshold:
        pm_module.RELIABILITY_THRESHOLD = float(rel_threshold) / 100.0
    if health_thresh:
        original_calc_device = pm_module.calculate_device_maintenance_info

        def patched_calc_device_maintenance_info(df, device_id, health_score):
            result = original_calc_device(df, device_id, health_score)
            result['强制优先'] = health_score < int(health_thresh)
            if health_score < int(health_thresh) and result.get('紧迫度') and result['紧迫度'] != '数据不足':
                from src.predictive_maintenance import get_urgency_level as _gul
                result['紧迫度'] = '紧急'
            return result

        pm_module.calculate_device_maintenance_info = patched_calc_device_maintenance_info

    max_date_str = str(df['日期'].max())
    health_scores = calculate_all_health_scores(df, max_date_str, DEFAULT_TAKT, 7)

    schedule_result = generate_full_maintenance_schedule(df, health_scores, horizon_days)
    SCHEDULE_STORE['schedule'] = schedule_result

    stats = schedule_result.get('统计信息', {})

    kpi_cards = [
        dbc.Col(get_kpi_card(
            "总设备数",
            str(stats.get('总设备数', 0)),
            f"数据充足: {stats.get('数据充足', 0)}台",
            "primary",
            "🔧"
        ), width=3),
        dbc.Col(get_kpi_card(
            "已分配排程",
            f"{stats.get('已分配', 0)}/{stats.get('总设备数', 0)}",
            f"未分配: {stats.get('未分配', 0)}台",
            "success" if stats.get('未分配', 0) == 0 else "warning",
            "✅"
        ), width=3),
        dbc.Col(get_kpi_card(
            "紧急设备",
            str(stats.get('紧急设备', 0)),
            "需立即关注并优先维护",
            "danger" if stats.get('紧急设备', 0) > 0 else "success",
            "⚠️"
        ), width=3),
        dbc.Col(get_kpi_card(
            "威布尔拟合",
            f"{stats.get('数据充足', 0)}台",
            f"数据不足: {stats.get('数据不足', 0)}台(使用固定周期)",
            "info",
            "📊"
        ), width=3),
    ]

    gantt_fig = create_maintenance_gantt_chart(schedule_result)
    schedule_list = create_schedule_list_table(schedule_result)
    weibull_panel = create_weibull_params_panel(schedule_result)

    return kpi_cards, gantt_fig, schedule_list, weibull_panel


@app.callback(
    Output('download-ics', 'data'),
    [Input('export-ics-btn', 'n_clicks')]
)
def export_ics_file(n_clicks):
    if not n_clicks or not SCHEDULE_STORE.get('schedule'):
        raise dash.exceptions.PreventUpdate

    ics_content = generate_ics_file(SCHEDULE_STORE['schedule'])
    if not ics_content:
        raise dash.exceptions.PreventUpdate

    current_time = SCHEDULE_STORE['schedule'].get('当前时间', datetime.now())
    filename = f"维保排程_{current_time.strftime('%Y%m%d')}.ics"

    return dcc.send_string(ics_content, filename=filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
