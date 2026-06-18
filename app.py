import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, date
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
    get_navbar(),
    html.Div(id='page-content', style={"paddingTop": "70px"}),
    get_footer(),
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
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
     Output('data-loaded-store', 'data')],
    [Input('upload-data', 'contents'),
     Input('load-sample-btn', 'n_clicks')],
    [State('upload-data', 'filename')]
)
def handle_file_upload(contents, n_clicks, filename):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return html.Div("请上传数据文件或加载示例数据", className="text-muted"), \
               html.Div("暂无数据", className="text-muted"), \
               html.Div("暂无数据", className="text-muted"), False
    
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
                   html.Div(""), html.Div(""), False
        
        upload_status = dbc.Alert("✅ 示例数据加载成功！", color="success")
    elif trigger_id == 'upload-data' and contents is not None:
        df, errors = parse_contents(contents, filename)
        if df is None:
            return dbc.Alert(f"❌ 文件上传失败: {errors[0] if errors else '未知错误'}", color="danger"), \
                   html.Div(""), html.Div(""), False
        
        if errors:
            upload_status = dbc.Alert(f"⚠️ 数据校验发现 {len(errors)} 个问题，请修正后重新上传", color="warning")
        else:
            upload_status = dbc.Alert("✅ 数据上传成功，校验通过！", color="success")
    else:
        return html.Div("请上传数据文件或加载示例数据", className="text-muted"), \
               html.Div("暂无数据", className="text-muted"), \
               html.Div("暂无数据", className="text-muted"), False
    
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
    
    return upload_status, error_list, preview, data_loaded


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
     Input('url', 'pathname')]
)
def update_overview_dates(data_loaded, pathname):
    return get_date_and_devices_from_store()


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
        
        content = html.Div([
            html.H5(f"设备: {device_name} — {factor_name}逐日明细", className="mb-2"),
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
     Input('anomaly-consecutive-days', 'value')]
)
def update_anomaly_detection(oee_target, drop_threshold, consecutive_days):
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
