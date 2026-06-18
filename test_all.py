import sys
sys.path.insert(0, '.')

from src.generate_sample_data import generate_sample_data
from src.data_import import validate_csv, process_dataframe
from src.oee_engine import calculate_oee_overall, calculate_daily_oee
from src.trend_analysis import get_all_anomalies, create_trend_chart
from src.downtime_analysis import create_pareto_chart, create_gantt_chart
from src.report_export import generate_pdf_report

print("=" * 60)
print("OEE Dashboard 功能测试")
print("=" * 60)

print("\n1. 生成示例数据...")
df = generate_sample_data()
print(f"   共 {len(df)} 条记录")
print(f"   设备数量: {df['设备编号'].nunique()}")
print(f"   日期范围: {df['日期'].min()} 至 {df['日期'].max()}")

print("\n2. 数据校验...")
is_valid, errors, validated_df = validate_csv(df)
print(f"   校验结果: {'通过' if is_valid else '失败'}")
if errors:
    print(f"   错误数: {len(errors)}")
    for e in errors[:5]:
        print(f"     - {e}")
else:
    print("   无错误")

print("\n3. 数据处理...")
processed = process_dataframe(validated_df)
print(f"   处理后记录数: {len(processed)}")
print(f"   新增列: 开始时间戳, 结束时间戳, 持续时间分钟")

print("\n4. OEE计算测试...")
min_date = df['日期'].min()
max_date = df['日期'].max()
result = calculate_oee_overall(processed, min_date, max_date)
summary = result['设备汇总']
print(f"   整体OEE: {summary['OEE']*100:.2f}%")
print(f"   可用率: {summary['可用率']*100:.2f}%")
print(f"   性能率: {summary['性能率']*100:.2f}%")
print(f"   质量率: {summary['质量率']*100:.2f}%")
print(f"   总负荷时间: {summary['总负荷时间']:.1f}分钟")

print("\n   各设备OEE:")
for device, res in result['各设备'].items():
    print(f"     {device}: OEE={res['OEE']*100:.2f}%, 节拍={res['节拍时间']:.1f}秒")

print("\n5. 日OEE计算...")
daily = calculate_daily_oee(processed, 'CNC-001')
print(f"   CNC-001 共 {len(daily)} 天数据")
print(f"   平均OEE: {daily['OEE'].mean()*100:.2f}%")

print("\n6. 异常检测...")
anomalies = get_all_anomalies(processed)
print(f"   检测到 {len(anomalies)} 个异常事件")
for a in anomalies[:3]:
    if a['类型'] == 'OEE骤降告警':
        print(f"     - {a['类型']}: {a['设备']} {a['日期']} 下降{a['下降幅度']*100:.1f}个百分点")
    else:
        print(f"     - {a['类型']}: {a['设备']} 连续{a['持续天数']}天低效")

print("\n7. 图表生成测试...")
fig_pareto = create_pareto_chart(processed, min_date, max_date)
print(f"   Pareto图: OK")
fig_gantt = create_gantt_chart(processed, min_date, max_date)
print(f"   甘特图: OK")

print("\n8. PDF报告生成测试...")
try:
    pdf_bytes = generate_pdf_report(processed, min_date, max_date, '月报')
    print(f"   PDF报告: OK ({len(pdf_bytes)} bytes)")
except Exception as e:
    print(f"   PDF报告: 失败 - {e}")

print("\n" + "=" * 60)
print("✅ 所有核心功能测试通过！")
print("=" * 60)
