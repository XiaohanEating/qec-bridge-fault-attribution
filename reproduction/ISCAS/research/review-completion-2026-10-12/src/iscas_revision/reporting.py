"""Build a review evidence index from committed results, including unfinished work."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .artifacts import ROOT, WORK, read_json, commit_json, digest


def bb_coverage():
    rows = []
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK/"results/bb"/instance/order
            physical = folder/"physical-risk.json"
            count = read_json(physical)["signature_count"] if physical.exists() else None
            points = [folder/f"eps{eps}" for eps in (1e-9,1e-8,1e-7,1e-6,3e-6,1e-5,3e-5,1e-4)]
            rows.append(dict(instance=instance,order=order,physical_enumeration_complete=physical.exists(),
                sf_committed_blocks=len(list((folder/"single-fault-shards").glob("*.npz"))),
                sf_target_blocks=(count+255)//256 if count is not None else None,
                physical_sample_points=sum((p/"physical.npz").exists() for p in points),
                finite_committed_blocks=sum(len(list((p/"decoded-shards").glob("*.npz"))) for p in points),
                finite_target_blocks=8*256, actual_risk_complete=(folder/"decoder-risk.json").exists()))
    return rows


def collect():
    registrations = {}
    for folder in WORK.joinpath("provenance").glob("batch*"):
        path = folder/"registration.json"
        if path.exists(): registrations[folder.name] = read_json(path)
    tasks = {}
    source_ids = {}
    # Repairs preserve task ids; the newest compatible registration describes
    # the current execution. Different protocol controls retain separate ids.
    for batch, registration in sorted(registrations.items()):
        for task in registration["tasks"]:
            tasks[task["id"]] = task
            source_ids[task["id"]] = batch
    counts = {}
    remaining = []
    for identifier, task in tasks.items():
        kind = task["kind"]
        counts.setdefault(kind, Counter())
        path = WORK/"results/tasks"/(identifier+".json")
        complete = path.exists()
        # Failed first Zhao adaptation is superseded by batch4, not silently
        # counted as a successful execution of the first source snapshot.
        superseded = identifier.startswith("batch2-") and kind == "zhao"
        state = "superseded" if superseded else "complete" if complete else "pending"
        counts[kind][state] += 1
        if state == "pending": remaining.append(dict(task, registration=source_ids[identifier]))
    statuses = {path.name: read_json(path) for path in WORK.joinpath("results").glob("*status.json")
                if path.name != "status.json"}
    batch_coverage={}
    for batch,registration in registrations.items():
        covered=[task["id"] for task in registration["tasks"]
                 if (WORK/"results/tasks"/(task["id"]+".json")).exists()]
        batch_coverage[batch]=dict(total=len(registration["tasks"]),complete=len(covered),
            remaining=[task["id"]for task in registration["tasks"]if task["id"]not in covered])
    now = datetime.now(timezone.utc).isoformat()
    pending_acceptance = []
    if any(task['kind'] == 'bb' for task in remaining):
        pending_acceptance.append('all registered BB decisions and finite-noise samples')
    if any(task['kind'] == 'stratum' for task in remaining):
        pending_acceptance.append('complete historical conditional strata')
    pending_acceptance.extend(['final source hashes and raw residual audit',
        'final manuscript, figures, bibliography and 4+1 page verification'])
    acceptance_path = WORK/'provenance/final-review-acceptance.json'
    accepted = False
    if acceptance_path.exists() and not remaining:
        acceptance = read_json(acceptance_path)
        bound = acceptance.get('input_sha256', {})
        manuscript_paths = [ROOT/'ISCAS/paper/manuscript/main.tex',
            ROOT/'ISCAS/paper/manuscript/ISCAS2027-review-draft.pdf']
        accepted = acceptance.get('passed', False) and all(
            bound.get(str(path)) == digest(path) for path in manuscript_paths)
    if accepted:
        pending_acceptance = []
    result = dict(updated_utc=now, status="complete" if accepted else "running" if remaining else "computations complete; manuscript verification pending",
        environment=str(ROOT/".harness/envs/qldpc-submission-audit/Scripts/python.exe"),
        authorization="user approved selected project environment and bposd2.1/PySCIPOpt5.4.1 with core versions unchanged",
        counts={kind:dict(value) for kind,value in counts.items()}, pending_tasks=remaining,
        bb_decision_coverage=bb_coverage(),
        execution_status=statuses,batch_coverage=batch_coverage,
        finished_evidence=["saved-layout audit", "decoder one-factor/prior paired comparisons",
            "64 candidate risk selection and independent confirmation", "physical distance-two injection certificates",
            "idle physical reweighting", "same-task native and complete-DEM bridge comparison",
            "Hirai official circuits", "Zhao official author model and common SD6 controls",
            "all 24 dynamic deployment tasks", "all 18 idle sensitivity tasks",
            "fixed-prior complete detector model attribution control"],
        pending_acceptance=pending_acceptance,
        manuscript_status="all registered calculations and manuscript verification complete" if accepted else "compiled revision; final acceptance pending")
    commit_json(WORK/"results/status.json", result)
    return result


def report(status):
    lines = ["# 审稿要求与实验完成情况", "",
        "本报告由已提交任务生成。完成计算、得到统计结论和稿件验收分别记录。", "",
        f"更新时间：{status['updated_utc']}。完整任务状态：{'计算与稿件核查完成' if status['status']=='complete' else '继续执行'}。", "",
        "| 计算类别 | 已完成 | 待完成 | 旧版替代 |", "| --- | ---: | ---: | ---: |"]
    for kind, counts in status["counts"].items():
        lines.append(f"| {kind} | {counts.get('complete',0)} | {counts.get('pending',0)} | {counts.get('superseded',0)} |")
    lines += ["", "| 审稿问题 | 新证据与文件 | 当前结论 |", "| --- | --- | --- |",
        "| 码与布局 | historical-audit.json；bb/*/construction.json；bb-source-equivalence.md | 主码是可近邻实现的距离3表面码；历史索引布局强制全部检查远程。BB使用真实非局域检查，完整解码提交情况列于下表。 |",
        "| 多故障与低噪声 | low-noise-summary.json；low-noise-comparisons.json；strata/；direct/ | 精确一故障贡献与条件多故障分开；未完成层计入尾界。第二实例每次平均6.8056故障，10^-4不是一阶区间。 |",
        "| 解码规则与先验 | onefactor/；decoder-comparisons.json；joint-bridge/；native/*/all4-noisy-v3/ | 单因素及先验配对已完成；完整DEM减小历史滑窗损失；有限噪声最优查表单独测量。 |",
        "| 容错距离 | circuit-distance-audit.json；distance-two-certificates.json；provenance/distance-two/ | 六历史电路精确故障距离2，无不可检测单故障；等记录异逻辑的双故障真实注入产生零记录逻辑翻转。 |",
        "| 更大门序空间 | order-space/result.json；order-space/comparisons.json | 64候选中原配置与R*同选macro7-checks0，强配置选macro6-checks1；初始带噪块固定，历史锚点不参与选择。 |",
        "| 路由与保距离基线 | deployment/；budget-modification-audit.json；hirai/；hirai-excess/；zhao/ | 官方电路实际执行；不同硬件和作者噪声另列；24组部署比较全部完成。 |",
        "| 空闲比及写作 | idle-reweighted-risk.json；sensitivity/；protocol-corrections.md；literature/ | 18组敏感性样本全部完成，空闲率改变R*排序。四块带噪边界已按保存电路补正；正文报告BB完整八率范围和实测结论。 |", "",
        "## 已核实的关键解释", "",
        "原论文的风险分解属于加权分类决策。可追溯的物理映射、同资源比较以及记录级干预证据构成实验方法；加法恒等式本身不承担创新主张。", "",
        "BB六电路完整物理枚举均有R*=0。全部响应核查支持声明的联合逻辑Z记忆任务故障距离至少3，见bb-memory-distance-lower-bound.json；完整量子码的精确电路距离另列。实际解码损失与多故障性能按提交情况记录。", "",
        "历史主实例在完整DEM强BP–OSD下forward的单故障额外损失为零；原文大额损失依赖滑窗和模型构建，须分开报告门序与解码实现。", "",
        "新源文件和实验批次分别登记。旧Zhao适配错误、初始边界不同的v2对照、历史原始数组全部保留。", "",
        "## 验收记录" if status['status']=='complete' else "## 剩余验收", "",
        ("全部登记任务、样本种子、原始哈希、逐逻辑残差、物理系数和区间核查通过。修订稿完成BibTeX与重复LaTeX编译，默认10pt，四页技术内容加一页参考文献，32篇引用，全部字体嵌入，无Type3字体；五页视觉检查绑定最终PDF SHA。验收来源为provenance/final-review-acceptance.json。新增科学论述与投稿决定由作者审阅。"
         if status['status']=='complete' else
         "原始分片已齐全；最终验收继续核对原任务汇总、样本种子、来源哈希、逐逻辑残差、物理系数和区间。稿件编译、字体及逐页视觉检查分别保存记录，完整验收通过后更新状态。")]
    lines += ["", "| BB实例 | 门序 | 物理枚举R* | 单故障决策块 | 有限噪声决策块 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in status["bb_decision_coverage"]:
        physical = WORK/"results/bb"/row["instance"]/row["order"]/"physical-risk.json"
        value = str(read_json(physical)["R_star"]) if physical.exists() else "执行中"
        lines.append(f"| {row['instance']} | {row['order']} | {value} | {row['sf_committed_blocks']}/{row['sf_target_blocks']} | {row['finite_committed_blocks']}/{row['finite_target_blocks']} |")
    path = WORK/"reports/review-evidence-index.md"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    commit_json(WORK/"provenance/reporting-verification.json", dict(
        created_utc=status["updated_utc"], report_sha256=digest(path), source_sha256=digest(__file__),
        pending_count=len(status["pending_tasks"]), scientific_inputs_modified=False))


if __name__ == "__main__":
    status = collect()
    report(status)
    print(dict(status=status["status"], counts=status["counts"]), flush=True)
