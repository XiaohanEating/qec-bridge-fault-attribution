"""Build a review evidence index from committed results, including unfinished work."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .artifacts import ROOT, WORK, read_json, commit_json, digest


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
    now = datetime.now(timezone.utc).isoformat()
    result = dict(updated_utc=now, status="running" if remaining else "computations complete; manuscript verification pending",
        environment=str(ROOT/".harness/envs/qldpc-submission-audit/Scripts/python.exe"),
        authorization="user approved selected project environment and bposd2.1/PySCIPOpt5.4.1 with core versions unchanged",
        counts={kind:dict(value) for kind,value in counts.items()}, pending_tasks=remaining,
        execution_status=statuses,
        finished_evidence=["saved-layout audit", "decoder one-factor/prior paired comparisons",
            "64 candidate risk selection and independent confirmation", "physical distance-two injection certificates",
            "idle physical reweighting", "same-task native and complete-DEM bridge comparison",
            "Hirai official circuits", "Zhao official author model and common SD6 controls"],
        pending_acceptance=["all registered BB decisions and finite-noise samples", "complete historical conditional strata",
            "remaining dynamic deployment and idle sensitivity", "source hashes and raw residual audit",
            "final results, manuscript, figures, bibliography and 4+1 page verification"],
        manuscript_status="revision in progress; existing review-draft is provisional")
    commit_json(WORK/"results/status.json", result)
    return result


def report(status):
    lines = ["# 审稿要求与实验完成情况", "",
        "本报告由已提交任务生成。完成计算、得到统计结论和稿件验收分别记录。", "",
        f"更新时间：{status['updated_utc']}。完整任务状态：继续执行。", "",
        "| 计算类别 | 已完成 | 待完成 | 旧版替代 |", "| --- | ---: | ---: | ---: |"]
    for kind, counts in status["counts"].items():
        lines.append(f"| {kind} | {counts.get('complete',0)} | {counts.get('pending',0)} | {counts.get('superseded',0)} |")
    lines += ["", "| 审稿问题 | 新证据与文件 | 当前结论 |", "| --- | --- | --- |",
        "| 码与布局 | historical-audit.json；bb/*/construction.json；bb-source-equivalence.md | 主码是可近邻实现的距离3表面码；历史索引布局强制全部检查远程。BB使用真实非局域检查，完整解码仍执行。 |",
        "| 多故障与低噪声 | low-noise-summary.json；low-noise-comparisons.json；strata/；direct/ | 精确一故障贡献与条件多故障分开；未完成层计入尾界。第二实例每次平均6.8056故障，10^-4不是一阶区间。 |",
        "| 解码规则与先验 | onefactor/；decoder-comparisons.json；joint-bridge/；native/*/all4-noisy-v3/ | 单因素及先验配对已完成；完整DEM减小历史滑窗损失；有限噪声最优查表单独测量。 |",
        "| 容错距离 | circuit-distance-audit.json；distance-two-certificates.json；provenance/distance-two/ | 六历史电路精确故障距离2，无不可检测单故障；等记录异逻辑的双故障真实注入产生零记录逻辑翻转。 |",
        "| 更大门序空间 | order-space/result.json；order-space/comparisons.json | 64候选中原配置与R*同选macro7-checks0，强配置选macro6-checks1；初始带噪块固定，历史锚点不参与选择。 |",
        "| 路由与保距离基线 | deployment/；budget-modification-audit.json；hirai/；hirai-excess/；zhao/ | 官方电路实际执行；不同硬件和作者噪声另列；646基线及全部有限噪声部署仍执行。 |",
        "| 空闲比及写作 | idle-reweighted-risk.json；sensitivity/；protocol-corrections.md；literature/ | 空闲率改变R*排序；敏感性样本和稿件仍执行。全4块带噪边界已按保存电路补正。 |", "",
        "## 已核实的关键解释", "",
        "原论文的风险分解属于加权分类决策。可追溯的物理映射、同资源比较以及记录级干预证据构成实验方法；加法恒等式本身不承担创新主张。", "",
        "BB-72三门序及BB-108 forward的完整单故障最优风险为零。其余BB顺序尚未完成枚举，全部实际解码损失与多故障性能仍执行。不能把R*=0等同于有限噪声不失败或已证明完整电路距离。", "",
        "历史主实例在完整DEM强BP–OSD下forward的单故障额外损失为零；原文大额损失依赖滑窗和模型构建，须分开报告门序与解码实现。", "",
        "新源文件和实验批次分别登记。旧Zhao适配错误、初始边界不同的v2对照、历史原始数组全部保留。", "",
        "## 剩余验收", "",
        "全部登记任务完成后核对样本种子、原始哈希、逐逻辑残差、物理系数及区间，重写论文实验与讨论。最终编译须含BibTeX与重复LaTeX，检查4页技术内容加1页参考文献、字体嵌入、文件大小和逐页视觉结果。"]
    path = WORK/"reports/review-evidence-index.md"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    commit_json(WORK/"provenance/reporting-verification.json", dict(
        created_utc=status["updated_utc"], report_sha256=digest(path), source_sha256=digest(__file__),
        pending_count=len(status["pending_tasks"]), scientific_inputs_modified=False))


if __name__ == "__main__":
    status = collect()
    report(status)
    print(dict(status=status["status"], counts=status["counts"]), flush=True)
