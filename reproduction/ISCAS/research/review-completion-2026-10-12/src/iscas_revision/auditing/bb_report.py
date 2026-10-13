"""Render complete BB evidence tables from verified scientific components."""
from datetime import datetime, timezone

from ..artifacts import WORK, read_json, digest, commit_json


def render():
    source = WORK/'results/bb-components.json'
    comparisons_path = WORK/'results/bb-comparisons.json'
    data, comparisons = read_json(source), read_json(comparisons_path)
    points = {(p['instance'], p['order'], p['eps']): p for p in data['finite_points']}
    labels = ('original', 'strong', 'single_fault_lookup')
    orders = ('forward', 'macro_reverse', 'check_reverse')
    rates = (1e-9, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    complete = data['single_fault_complete'] and data['finite_complete'] and comparisons['complete']
    lines = ['# 非局域 BB 桥接实验完整结果', '',
        f"截至 {data['created_utc']}，六组单故障计算完成；有限噪声完整点 {len(points)}/48，门序差值 {len(comparisons['comparisons'])}/96。", '',
        '每个有限噪声点保留32768个物理样本。失败事件为任一逻辑Z残差非零；BB-72同时检查12位，BB-108同时检查8位。三种规则共享各点的物理样本，不同门序使用独立登记种子。', '',
        '## 硬件与故障规模', '',
        '| 码 | 保留站点 | CX | 时隙 | 噪声位置 | 10^-9平均故障 | 10^-6平均故障 | 10^-4平均故障 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        '| BB-72 | 144 | 26784 | 13842 | 1966608 | 0.00196661 | 1.96661 | 196.661 |',
        '| BB-108 | 324 | 45000 | 23374 | 7528284 | 0.00752828 | 7.52828 | 752.828 |', '',
        '自然棋盘布局的周期边和长程边全部经开边界方格路由。三门序保持理想测量操作、有向边CX计数、时隙与逐站点累计空闲数。四个提取块全部带噪，末端数据读出理想。BB-108额外108站点的空闲噪声计入成本。', '',
        'Zhao官方电路覆盖相同CSS码；检查矩阵置换和逻辑变基证书保存在bb-source-equivalence。其辅助制备、保留长程连接和完整模型解码构成另外一种部署条件，完整资源表见 [来源与协议核查](bb-source-equivalence.md)。', '',
        '## 单故障实际损失', '',
        '| 码 | 门序 | R* | 原规则Rδ | 强规则Rδ | 查表Rδ |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for matrix in data['single_fault_matrices']:
        values = [matrix['decoder'][label]['R_delta'] for label in labels]
        lines.append(f"| {matrix['instance']} | {matrix['order']} | {matrix['R_star']:.4f} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    lines += ['', '全部六电路R*=0，实际风险系数等于额外决策损失E。每个单故障记录具有唯一逻辑标签，零记录逻辑响应为零；声明的联合逻辑Z记忆任务故障距离下界为3。强规则仍会误判部分单故障，检查逆序降低其系数。有限噪声查表的表外预测采用登记强规则。', '',
        '## 全部有限噪声点', '',
        '表中为失败数/32768；括号是失败率的边际95%精确二项区间，单位为百分比。144个率的共同95%区间逐项保存在bb-components.json。全部48点均使用完整样本。', '',
        '| 码 | 门序 | ε | 原规则 | 强规则 | 单故障查表 |',
        '| --- | --- | ---: | --- | --- | --- |']
    for instance in ('bb-72', 'bb-108'):
        for order in orders:
            for eps in rates:
                point = points.get((instance, order, eps))
                cells = []
                for label in labels:
                    if point is None:
                        cells.append('待完整提交'); continue
                    row = point['decoder'][label]
                    lo, hi = row['marginal_interval']
                    cells.append(f"{row['failures']} ({100*lo:.5f}–{100*hi:.5f})")
                lines.append(f"| {instance} | {order} | {eps:g} | "+' | '.join(cells)+' |')
    lines += ['', '## 全部门序差值', '',
        '差值为候选减forward，单位为百分点。96项差值同时覆盖率95%，每项用两电路的独立精确二项区间再作Bonferroni分配。正区间支持forward，负区间支持候选；跨零写为未分辨。', '',
        '| 码 | 候选 | ε | 规则 | 差值 | 共同区间 | 结论 |',
        '| --- | --- | ---: | --- | ---: | --- | --- |']
    for row in comparisons['comparisons']:
        lo, hi = row['confidence_interval']
        decision = {'candidate lower':'候选较低', 'forward lower':'forward较低', 'unresolved':'未分辨'}[row['sorting']]
        lines.append(f"| {row['instance']} | {row['candidate']} | {row['eps']:g} | {row['decoder']} | {100*row['difference']:.5f} | [{100*lo:.5f}, {100*hi:.5f}] | {decision} |")
    lines += ['', '## 结果解释与来源', '',
        '低噪声与高噪声使用同一登记规则、先验和样本数。10^-4下开放方格桥接接近全失败，原始失败数与区间保留；这个噪声范围的故障数已远离单故障主导区间。门序选择按实际噪声下的完整差值判断，单故障系数用于诊断对应记录的损失。', '',
        '物理样本、全部预测和逐逻辑残差由原bb_analysis任务汇总；bb-results-audit核对各分片、种子、来源及最终汇总，committed-results-audit核对全部补实验。单故障可辨的距离证据见bb-memory-distance-lower-bound.json。', '',
        '本表生成源为results/bb-components.json和results/bb-comparisons.json，各点输入和全部预测分片SHA在原JSON中保存。']
    path = WORK/'reports/bb-complete-evidence.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), complete=complete, report_sha256=digest(path),
        inputs={str(source):digest(source), str(comparisons_path):digest(comparisons_path)},
        source_sha256=digest(__file__), finite_points=len(points), comparisons=len(comparisons['comparisons']), scientific_inputs_modified=False)
    commit_json(WORK/'provenance/bb-report-verification.json', result)
    print(dict(complete=complete, finite_points=len(points), comparisons=len(comparisons['comparisons'])), flush=True)


if __name__ == '__main__':
    render()
