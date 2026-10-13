"""Read-only statistical synthesis, with comparison-wide coverage stated explicitly."""
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import stim
from scipy.stats import binom
from .artifacts import ROOT, WORK, HISTORICAL, read_json, commit_json, digest
from .audit import optimal_table, paired_comparison
from .stratification import integrate, clopper_pearson

ORDERS = ("forward", "macro_reverse", "check_reverse")
INSTANCES = ("tiny-hgp", "holdout-hgp-3x4")
DECODERS = ("original", "strong", "single_fault_lookup")
EPS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)


def independent_difference(first, second, alpha=.05):
    """Candidate minus baseline: two CP intervals at 1-alpha/2 coverage."""
    alo, ahi = clopper_pearson(first["failures"], first["shots"], alpha/2)
    blo, bhi = clopper_pearson(second["failures"], second["shots"], alpha/2)
    return dict(difference=second["failures"]/second["shots"]-first["failures"]/first["shots"],
                confidence_interval=[blo-ahi, bhi-alo], confidence_level=1-alpha,
                method="Bonferroni two independent Clopper-Pearson intervals")


def holm(rows, family_size):
    """Unfinished tests occupy the remainder of a prespecified family."""
    assert len(rows) <= family_size
    maximum = 0.
    for rank, row in enumerate(sorted(rows, key=lambda row: row["mcnemar_exact_two_sided_p"])):
        maximum = max(maximum, min(1., (family_size-rank)*row["mcnemar_exact_two_sided_p"]))
        row["holm_adjusted_p"] = maximum
    return rows


def decoder_comparisons():
    single = []
    prior = []
    for instance in INSTANCES:
        for order in ORDERS:
            folder = WORK/"results/onefactor"/instance/order
            records = {}
            for setting in ("fixed", "matched"):
                path = folder/setting/"result.json"
                if not path.exists():
                    continue
                data = read_json(path)
                records[setting] = data
                for name, value in data["paired"].items():
                    single.append(dict(instance=instance, order=order, prior=setting,
                                       configuration=name, **value))
            if len(records) == 2:
                assert records["fixed"]["sample_sha256"] == records["matched"]["sample_sha256"]
                a = np.load(folder/"fixed/decoded.npz")
                b = np.load(folder/"matched/decoded.npz")
                for name in records["fixed"]["results"]:
                    prior.append(dict(instance=instance, order=order, configuration=name,
                        **paired_comparison(a[name+"_residual"], b[name+"_residual"])))
    result = dict(single_factor_family_size=96, single_factor_tests=holm(single,96),
                  prior_family_size=54, prior_tests=holm(prior,54),
                  status="complete" if len(single)==96 and len(prior)==54 else "partial",
                  interpretation="Paired reanalysis of the saved 4096-shot exploration; no new independent samples")
    commit_json(WORK/"results/decoder-comparisons.json", result)
    return result


def idle_reweighting():
    rows = []
    for order in ORDERS:
        folder = HISTORICAL/"results/tiny-hgp"/order
        circuit = stim.Circuit((HISTORICAL/"provenance/decoder-priors/tiny-hgp"/(order+".stim")).read_text(encoding="utf-8"))
        source = read_json(folder/"branches.json")
        data = np.load(folder/"single-fault.npz")
        idle_offsets = {}
        active_h = set()
        channel_counts = Counter()
        for offset, inst in enumerate(circuit.flattened()):
            if inst.name == "TICK":
                active_h = set()
            elif inst.name == "H":
                active_h.update(target.qubit_value for target in inst.targets_copy())
            elif inst.name == "DEPOLARIZE1":
                idle = {target.qubit_value for target in inst.targets_copy()}-active_h
                idle_offsets[offset] = idle
                channel_counts["idle"] += len(idle)
                channel_counts["single_qubit_gate"] += len(inst.targets_copy())-len(idle)
        construction = read_json(folder.parent/"construction.json")
        assert channel_counts["idle"] == construction[order]["metrics"]["noisy_idle_locations"]
        uniform, optimum_uniform = optimal_table(data["detectors"], data["observables"], data["physical_coefficients"])
        uniform_pred = np.array([uniform[bytes(d)] for d in data["detectors"]])
        for rho in (0., .1, 1.):
            mass = np.zeros(len(data["detectors"]))
            idle_mass = np.zeros_like(mass)
            gate_mass = np.zeros_like(mass)
            for branch in source["branches"]:
                q = branch["paulis"][0][0]
                is_idle = q in idle_offsets.get(branch["instruction_offset"], set())
                weight = branch["coefficient"]
                (idle_mass if is_idle else gate_mass)[branch["signature"]] += weight
            mass = gate_mass + rho*idle_mass
            table, optimum = optimal_table(data["detectors"], data["observables"], mass)
            predictions = {name:data[name] for name in ("original","strong")}
            predictions["fixed_SD6_lookup"] = uniform_pred
            risks = {}
            for name, pred in predictions.items():
                failed = (pred ^ data["observables"]).any(axis=1)
                risk = float(mass @ failed)
                risks[name] = dict(R_delta=risk, E=risk-optimum,
                    gate_contribution=float(gate_mass @ failed),
                    scaled_idle_contribution=float(rho*idle_mass @ failed))
                assert risk >= optimum-1e-7
            if rho == 1:
                assert np.allclose(mass, data["physical_coefficients"])
                assert abs(optimum-optimum_uniform) < 1e-7
            rows.append(dict(order=order, rho=rho, R_star=optimum, decoder=risks,
                        visible_gate_mass=float(gate_mass.sum()), visible_idle_mass=float(idle_mass.sum()),
                        noise_channel_counts=dict(channel_counts),
                        branch_sha256=digest(folder/"branches.json"),
                        note="R_star reoptimizes logical labels at each rho; finite-noise sensitivity uses the fixed SD6 lookup"))
    commit_json(WORK/"results/idle-reweighted-risk.json", rows)
    return rows


def low_noise_comparisons():
    """95% simultaneous coverage for the 60 registered candidate comparisons."""
    family_size = len(INSTANCES)*2*len(EPS)*len(DECODERS)
    summaries = {}
    for instance in INSTANCES:
        for order in ORDERS:
            old = read_json(HISTORICAL/"results"/instance/order/"risk.json")
            n = int(old["total_physical_coefficient"])
            kmax = int(binom.ppf(1-1e-8, n, max(EPS)))
            strata = {}
            for k in range(2,kmax+1):
                path = WORK/"results/strata"/instance/order/f"k{k:02d}"/"result.json"
                if not path.exists():
                    break
                strata[k] = read_json(path)
            coefficients = {name:old["decoder"][name]["coefficient"] for name in ("original","strong")}
            coefficients["single_fault_lookup"] = old["optimal_single_fault_coefficient"]
            for eps in EPS:
                for name in DECODERS:
                    records = {k:dict(shots=value["shots"],failures=value["failures"][name]) for k,value in strata.items()}
                    summaries[instance,order,eps,name] = dict(
                        marginal=integrate(n,eps,coefficients[name],records),
                        simultaneous=integrate(n,eps,coefficients[name],records,alpha=.05/(2*family_size)),
                        complete=len(strata)==kmax-1, completed_kmax=max(strata,default=1), target_kmax=kmax)
    rows = []
    for instance in INSTANCES:
        for candidate in ORDERS[1:]:
            for eps in EPS:
                for name in DECODERS:
                    a = summaries[instance,"forward",eps,name]
                    b = summaries[instance,candidate,eps,name]
                    alo,ahi = a["simultaneous"]["confidence_interval"]
                    blo,bhi = b["simultaneous"]["confidence_interval"]
                    ci = [blo-ahi,bhi-alo]
                    rows.append(dict(instance=instance, candidate=candidate, eps=eps, decoder=name,
                        difference=b["marginal"]["probability"]-a["marginal"]["probability"],
                        simultaneous_difference_interval=ci,
                        sorting="candidate lower" if ci[1]<0 else "forward lower" if ci[0]>0 else "unresolved",
                        complete=a["complete"] and b["complete"],
                        forward=a["marginal"],candidate_result=b["marginal"]))
    result = dict(family_size=family_size, simultaneous_confidence_level=.95,
        method="Bonferroni across 60 differences and two circuit intervals, then across conditional layers; explicit tail bounds",
        comparisons=rows)
    commit_json(WORK/"results/low-noise-comparisons.json",result)
    return result


def order_space_comparisons():
    result = read_json(WORK/"results/order-space/result.json")
    rows = []
    for sample in result["samples"]:
        if sample["task"]["order"]=="anchor-forward":
            continue
        baseline = next(x for x in result["samples"] if x["task"]["order"]=="anchor-forward"
                        and x["task"]["eps"]==sample["task"]["eps"])
        for name in DECODERS:
            rows.append(dict(order=sample["task"]["order"],eps=sample["task"]["eps"],decoder=name,
                baseline=baseline["decoder"][name],candidate=sample["decoder"][name],
                **independent_difference(baseline["decoder"][name],sample["decoder"][name], alpha=.05/12)))
    commit_json(WORK/"results/order-space/comparisons.json",dict(
        family_size=12, simultaneous_confidence_level=.95, comparisons=rows,
        winners=result["winners"],
        note="All original/optimal winners coincide; configuration-specific strong winner differs"))
    return rows


def deployment_and_sensitivity():
    """Independent circuit comparisons use declared families and exact intervals."""
    deployment=[]
    family=60  # Five candidates, four rates, three decision rules.
    for candidate in ("region1","kutin","forward","macro_reverse","check_reverse"):
        for eps in (1e-6,1e-5,1e-4,.001):
            baseline=WORK/"results/deployment/dynamic646"/f"eps{eps}"/"result.json"
            target=WORK/"results/deployment"/candidate/f"eps{eps}"/"result.json"
            if not baseline.exists() or not target.exists(): continue
            a,b=read_json(baseline),read_json(target)
            for name in DECODERS:
                deployment.append(dict(candidate=candidate,eps=eps,decoder=name,
                    baseline=a["decoder"][name],candidate_result=b["decoder"][name],
                    baseline_metrics=a["metrics"],candidate_metrics=b["metrics"],
                    **independent_difference(a["decoder"][name],b["decoder"][name],alpha=.05/family)))
    commit_json(WORK/"results/deployment-comparisons.json",dict(family_size=family,
        simultaneous_confidence_level=.95,comparisons=deployment,
        note="Absolute deployment costs differ; only region1 versus dynamic646 has matching resources"))
    sensitivity=[]
    family=36  # Two candidates, three idle ratios, two rates, three rules.
    for candidate in ORDERS[1:]:
        for rho in (0.,.1,1.):
            for eps in (1e-5,1e-4):
                leaf=f"rho{rho}-eps{eps}/result.json"
                baseline=WORK/"results/sensitivity/tiny-hgp/forward"/leaf
                target=WORK/"results/sensitivity/tiny-hgp"/candidate/leaf
                if not baseline.exists() or not target.exists(): continue
                a,b=read_json(baseline),read_json(target)
                for name in DECODERS:
                    sensitivity.append(dict(candidate=candidate,rho=rho,eps=eps,decoder=name,
                        baseline=a["decoder"][name],candidate_result=b["decoder"][name],
                        **independent_difference(a["decoder"][name],b["decoder"][name],alpha=.05/family)))
    commit_json(WORK/"results/sensitivity-comparisons.json",dict(family_size=family,
        simultaneous_confidence_level=.95,comparisons=sensitivity,
        note="Gate rate fixed; idle rate rho times gate rate; fixed full-SD6 lookup and decoder priors"))
    return dict(deployment=len(deployment),sensitivity=len(sensitivity))


def circuit_distance_audit():
    """Single faults distinguish uncorrectable ambiguity from distance-one faults."""
    rows=[]
    for instance in INSTANCES:
        for order in ORDERS:
            source=HISTORICAL/"results"/instance/order/"single-fault.npz"
            data=np.load(source)
            d,l=data["detectors"],data["observables"]
            zero=(d==0).all(axis=1)&l.any(axis=1)
            seen={}; witness=None
            for index,(record,label) in enumerate(zip(d,l)):
                key=bytes(record)
                if key in seen and not np.array_equal(l[seen[key]],label):
                    witness=[seen[key],index];break
                seen[key]=index
            assert not zero.any() and witness is not None
            # Two distinct labels for the same record certify an undetectable
            # logical product of two faults, while no one-fault product exists.
            rows.append(dict(instance=instance,order=order,exact_fault_distance=2,
                one_fault_zero_detector_logical_count=0,pair_signature_indices=witness,
                common_detectors=np.flatnonzero(d[witness[0]]).tolist(),
                logical_difference=(l[witness[0]]^l[witness[1]]).tolist(),source_sha256=digest(source),
                explanation="Every physical single-fault response was enumerated; none has D=0,L!=0; an equal-D unequal-L pair exists"))
    for distance in (3,5):
        source=WORK/"results/native"/f"d{distance}"/"all4-noisy-v3/eps0.0001/circuit.stim"
        circuit=stim.Circuit(source.read_text(encoding="utf-8"))
        rows.append(dict(instance=f"native-d{distance}",graphlike_distance=len(circuit.shortest_graphlike_error()),
                    R_star=read_json(source.parent/"result.json")["R_star"],source_sha256=digest(source),
                    explanation="Stim graphlike search; general hyperedge distance kept distinct"))
    commit_json(WORK/"results/circuit-distance-audit.json",rows)
    return rows


def run():
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  idle_reweighting_rows=len(idle_reweighting()),
                  decoder_status=decoder_comparisons()["status"],
                  low_noise_comparisons=len(low_noise_comparisons()["comparisons"]),
                  order_space_comparisons=len(order_space_comparisons()),
                  deployment_sensitivity=deployment_and_sensitivity(),
                  circuit_distance_rows=len(circuit_distance_audit()),
                  scientific_inputs_modified=False, source_sha256=digest(__file__))
    commit_json(WORK/"provenance/synthesis-verification.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    run()
