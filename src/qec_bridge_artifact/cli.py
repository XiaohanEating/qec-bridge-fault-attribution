"""Fetch, verify, audit, and replay the published reproduction artifact."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
TREE = ROOT / "reproduction"
WORK = TREE / "ISCAS/research/review-completion-2026-10-12"
HIST = TREE / "ISCAS/research/mechanism-intervention-2026-10-11"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def activate(output=None):
    sys.path.insert(0, str(WORK / "src"))
    os.environ["ISCAS_REVIEW_WORKDIR"] = str(WORK)
    artifacts = importlib.import_module("iscas_revision.artifacts")
    artifacts.ROOT = TREE
    artifacts.HISTORICAL = HIST
    artifacts.WORK = Path(output).resolve() if output else WORK
    return artifacts


def checked_target(name):
    target = (ROOT / name).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Archive path escapes artifact: {name}")
    return target


def fetch(args):
    assets = read(ROOT / "data/archive-manifest.json")
    selected = assets if args.all else [a for a in assets if a["name"] == "core-data.tar.gz"]
    download = Path(args.from_dir).resolve() if args.from_dir else ROOT / "downloads"
    download.mkdir(parents=True, exist_ok=True)
    for asset in selected:
        path = download / asset["name"]
        if not path.exists():
            if args.from_dir:
                raise FileNotFoundError(path)
            temporary = path.with_suffix(path.suffix + ".part")
            print(f"Downloading {asset['name']} ({asset['bytes']} bytes)", flush=True)
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "qec-bridge-artifact/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1024 * 1024)
            if sha(temporary) != asset["sha256"]:
                raise ValueError(f"Checksum mismatch: {temporary}")
            temporary.replace(path)
        if path.stat().st_size != asset["bytes"] or sha(path) != asset["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        expected = {entry["path"]: entry for entry in asset["files"]}
        seen = set()
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                seen.add(member.name)
                entry = expected[member.name]
                target = checked_target(member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha(target) == entry["sha256"]:
                    continue
                if target.exists():
                    raise FileExistsError(f"Existing artifact differs: {target}")
                temporary = target.with_name(target.name + ".extracting")
                with archive.extractfile(member) as source, temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest, 1024 * 1024)
                if temporary.stat().st_size != entry["bytes"] or sha(temporary) != entry["sha256"]:
                    raise ValueError(f"Member checksum mismatch: {member.name}")
                temporary.replace(target)
        if seen != set(expected):
            raise ValueError(f"Incomplete archive: {asset['name']}")
        print(f"Verified and restored {asset['name']}: {len(seen)} files", flush=True)


def verify(args):
    records = read(ROOT / "provenance/source-manifest.json")
    if args.data:
        records += [entry for asset in read(ROOT / "data/archive-manifest.json") for entry in asset["files"]]
    for index, entry in enumerate(records, start=1):
        path = checked_target(entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Missing or changed artifact: {entry['path']}")
        if index % 5000 == 0:
            print(f"Verified {index}/{len(records)} files", flush=True)
    print(json.dumps({"passed": True, "files": len(records), "data": args.data}))


def audit(args):
    import numpy as np
    activate()
    from iscas_revision.audit import optimal_table
    counts = Counter()
    risks = []
    for instance in ("tiny-hgp", "holdout-hgp-3x4"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = HIST / "results" / instance / order
            with np.load(folder / "single-fault.npz") as data:
                _, optimum = optimal_table(data["detectors"], data["observables"], data["physical_coefficients"])
                expected = read(folder / "risk.json")
                assert abs(optimum - expected["optimal_single_fault_coefficient"]) < 1e-7
                for rule in ("original", "strong"):
                    risk = float(data["physical_coefficients"] @ (data[rule] ^ data["observables"]).any(axis=1))
                    assert abs(risk - expected["decoder"][rule]["coefficient"]) < 1e-7
                risks.append({"instance": instance, "order": order, "R_star": optimum})
                counts["historical_risks"] += 1
    for instance in ("bb-72", "bb-108"):
        for order in ("forward", "macro_reverse", "check_reverse"):
            folder = WORK / "results/bb" / instance / order
            with np.load(folder / "single-fault.npz") as data, np.load(folder / "single-fault-decoded.npz") as decoded:
                _, optimum = optimal_table(data["detectors"], data["observables"], data["physical_coefficients"])
                expected = read(folder / "decoder-risk.json")
                assert abs(optimum - expected["R_star"]) < 1e-7
                for rule in expected["decoder"]:
                    risk = float(data["physical_coefficients"] @ (decoded[rule] ^ data["observables"]).any(axis=1))
                    assert abs(risk - expected["decoder"][rule]["R_delta"]) < 1e-6
                counts["bb_risks"] += 1
    for record_path in sorted((WORK / "results").rglob("result.json")):
        folder = record_path.parent
        physical, decoded = folder / "physical.npz", folder / "decoded.npz"
        if not decoded.exists():
            continue
        record = read(record_path)
        if not physical.exists() and "configurations" in record:
            physical = HIST / "results" / record["instance"] / record["order"] / "samples-v2/all-eps0.0001-n4096/physical-samples.npz"
        if not physical.exists() and "sample_source" in record:
            normalized = record["sample_source"].replace("\\", "/")
            physical = TREE / normalized.split("1014gogogo/", 1)[1]
        if not physical.exists():
            raise FileNotFoundError(f"Physical samples missing for {record_path}")
        if "sample_sha256" in record:
            assert sha(physical) == record["sample_sha256"]
        if "decoded_sha256" in record:
            assert sha(decoded) == record["decoded_sha256"]
        with np.load(physical) as samples, np.load(decoded) as predictions:
            observed = samples["observables"]
            rows = record.get("decoder", record.get("results", {}))
            for key in predictions.files:
                if not key.endswith("_prediction") or "single_fault_prediction" in key:
                    continue
                rule = key[:-11]
                residual = predictions[rule + "_residual"]
                assert np.array_equal(predictions[key] ^ observed, residual)
                failures = int(residual.any(axis=1).sum())
                if rule in rows:
                    assert failures == rows[rule]["failures"]
                elif isinstance(record.get("failures"), dict):
                    assert failures == record["failures"][rule]
                elif rule == "matching":
                    assert failures == record["failures"]
                else:
                    raise ValueError(f"Unbound decoder: {record_path}, {rule}")
                counts["decision_arrays"] += 1
            counts["sample_records"] += 1
            if counts["sample_records"] % 1000 == 0:
                print(f"Audited {counts['sample_records']} sample records", flush=True)
    result = {"passed": True, "counts": dict(counts), "historical_risks": risks}
    write(Path(args.output) / "audit.json", result)
    print(json.dumps(result))


def summarize(args):
    output = Path(args.output)
    primary = []
    for order in ("forward", "macro_reverse", "check_reverse"):
        old = read(HIST / "results/tiny-hgp" / order / "risk.json")
        complete = read(WORK / "results/joint-bridge" / order / "eps0.0001/result.json")
        primary.append({"order": order, "R_star": old["optimal_single_fault_coefficient"],
                        "original": old["decoder"]["original"]["coefficient"],
                        "strong": old["decoder"]["strong"]["coefficient"],
                        "complete_strong": complete["decoder"]["full_joint_strong"]["R_delta"]})
    bb = []
    for order in ("forward", "macro_reverse", "check_reverse"):
        bb.append({"order": order, **{name: read(WORK / "results/bb" / name / order / "decoder-risk.json")["decoder"]["strong"]["E"] for name in ("bb-72", "bb-108")}})
    write(output / "paper-tables.json", {"table1": primary, "table2": bb})
    for table, rows in (("table1", primary), ("table2", bb)):
        import csv
        with (output / f"{table}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    activate()
    from iscas_revision import figures
    scratch = output / "figure-replay"
    target = scratch / "results/order-space"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "comparisons.json"):
        shutil.copy2(WORK / "results/order-space" / name, target / name)
    figures.WORK = scratch
    figures.order_space()
    print(f"Tables and Figure 2 regenerated in {output}")


def upstream(args):
    target = Path(args.target).resolve() if args.target else WORK / "provenance/external"
    for entry in read(ROOT / "provenance/upstream.json"):
        folder = target / entry["name"]
        if not folder.exists():
            subprocess.run(["git", "clone", "--no-checkout", "--config", "core.longpaths=true", entry["repository"], str(folder)], check=True)
        subprocess.run(["git", "-c", "core.longpaths=true", "-c", f"safe.directory={folder.as_posix()}", "-C", str(folder), "checkout", "--detach", entry["commit"]], check=True)
        commit = subprocess.check_output(["git", "-c", f"safe.directory={folder.as_posix()}", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        if commit != entry["commit"]:
            raise ValueError("Wrong upstream source revision")
        print(f"{entry['name']}: {commit}")


def tasks(args):
    entries = read(ROOT / "data/registered-tasks.json")
    if args.kind:
        entries = [t for t in entries if t["kind"] == args.kind]
    print(json.dumps(entries, indent=2))


def replay(args):
    entries = read(ROOT / "data/registered-tasks.json")
    task = next(t for t in entries if t["id"] == args.id)
    output = Path(args.output).resolve()
    if output == WORK.resolve() or output.is_relative_to(WORK.resolve()):
        raise ValueError("Replay output must be outside the published inputs")
    activate(output)
    if task["kind"] in {"hirai", "hirai_excess", "zhao"}:
        if not (WORK / "provenance/external").exists():
            raise FileNotFoundError("Run fetch-upstream first")
        # Upstream inputs are read from their pinned original tree.
        copy_target = output / "provenance/external"
        if not copy_target.exists():
            shutil.copytree(WORK / "provenance/external", copy_target, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    if task["kind"] == "bb":
        source = WORK / "results/bb" / task["instance"] / task["order"]
        dest = output / "results/bb" / task["instance"] / task["order"]
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("prior.stim", "branches.json", "single-fault.npz", "physical-risk.json"):
            shutil.copy2(source / name, dest / name)
    from iscas_revision.experiments import dispatch
    print(json.dumps(dispatch(task)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("fetch-data"); p.add_argument("--all", action="store_true"); p.add_argument("--from-dir"); p.set_defaults(func=fetch)
    p = subs.add_parser("verify"); p.add_argument("--data", action="store_true"); p.set_defaults(func=verify)
    p = subs.add_parser("audit"); p.add_argument("--output", default="outputs"); p.set_defaults(func=audit)
    p = subs.add_parser("summarize"); p.add_argument("--output", default="outputs"); p.set_defaults(func=summarize)
    p = subs.add_parser("fetch-upstream"); p.add_argument("--target"); p.set_defaults(func=upstream)
    p = subs.add_parser("tasks"); p.add_argument("--kind"); p.set_defaults(func=tasks)
    p = subs.add_parser("replay"); p.add_argument("--id", required=True); p.add_argument("--output", required=True); p.set_defaults(func=replay)
    args = parser.parse_args()
    args.func(args)
