"""All three noisy-round reversal decisions, with a registered selection rule."""
from collections import Counter
import sys
import time
import numpy as np
import stim
from scipy.optimize import brentq
from scipy.stats import binom
from .artifacts import ROOT, WORK, HISTORICAL, read_json, commit_json, digest
from .audit import optimal_table
from .decoding import CachedDecoder
from .experiments import save_npz,decode_sample,rescale_noise
from construction import inherited,synthesize,match_budget,primitive_multiset
from physical_risk import physical_branches
from support import code_for


def finite_noise_certificates():
    rows=[]
    for instance in ("tiny-hgp","holdout-hgp-3x4"):
        forward=read_json(HISTORICAL/"results"/instance/"forward/risk.json")
        n=int(forward["total_physical_coefficient"])
        for order in ("macro_reverse","check_reverse"):
            candidate=read_json(HISTORICAL/"results"/instance/order/"risk.json")
            deltas={name:candidate["decoder"][name]["coefficient"]-forward["decoder"][name]["coefficient"]
                    for name in ("original","strong")}
            deltas["single_fault_lookup"]=candidate["optimal_single_fault_coefficient"]-forward["optimal_single_fault_coefficient"]
            for name,delta in deltas.items():
                def margin(eps): return binom.pmf(1,n,eps)*abs(delta)/n-binom.sf(1,n,eps)
                radius=brentq(margin,1e-14,.01,xtol=1e-15) if delta else 0.
                bounds={}
                for eps in (1e-6,3e-6,1e-5,3e-5,1e-4):
                    single=float(binom.pmf(1,n,eps)*delta/n)
                    multiple=float(binom.sf(1,n,eps))
                    bounds[str(eps)]=dict(difference_interval=[single-multiple,single+multiple],
                        ordering_certified="candidate improves" if single+multiple<0 else "candidate worsens" if single-multiple>0 else "multiple-fault sampling needed")
                rows.append(dict(instance=instance,candidate=order,decoder=name,delta_R=delta,
                       certified_noise_radius=radius,exact_deterministic_bounds=bounds))
    result=dict(proof="Matched circuits have the same N independent equal-rate fault locations. With delta(0)=0, exact one-fault contribution to candidate-minus-forward failure is Pr(K=1)*delta_R/N. The remaining contribution is Pr(K>=2)*(q_candidate-q_forward), bounded by +/-Pr(K>=2). This is a finite-noise sign certificate, without an unknown big-O constant.",
        formula="Delta P belongs to [epsilon*(1-epsilon)^(N-1)*Delta R - Pr(K>=2), epsilon*(1-epsilon)^(N-1)*Delta R + Pr(K>=2)]",
        rows=rows)
    commit_json(WORK/"results/finite-noise-certificates.json",result)
    return result


def compile_round_policy(original,placement,macro_mask,check_mask):
    blocks=[]
    def block(gates,mapping,side,axis,order):
        index=len(blocks)
        macro=bool(macro_mask & (1 << (index-1))) if index else False
        check=bool(check_mask & (1 << (index-1))) if index else False
        chosen=list(reversed(gates)) if check else list(gates)
        primitives=[]
        for a,b in chosen:
            path=inherited.path_between(mapping[a],mapping[b],side,"row")
            remote=synthesize(path,"bridge")
            primitives.extend(reversed(remote) if macro else remote)
        layers=inherited.pack_primitives(primitives)
        inherited.check_block(gates,layers,mapping,mapping,side)
        blocks.append(dict(index=index,noisy=True,macro_reverse=macro,check_reverse=check,signed_tableau_equal=True))
        return layers
    previous=inherited.restored_block;inherited.restored_block=block
    try: plan=inherited.compile_plan(original,placement,"restored","row","short")
    finally: inherited.restored_block=previous
    assert len(blocks)==4
    plan["policy"]=dict(macro_mask=macro_mask,check_mask=check_mask,blocks=blocks,initial_round_fixed=True)
    return plan


def execute(task):
    folder=WORK/"results/order-space"
    registration=folder/"registration.json"
    if not registration.exists():
        commit_json(registration,dict(code="tiny H3xH3",rounds=3,candidates=64,
            design="independent binary macro and check reversal for each of three repeat extraction blocks; initial noisy block fixed",
            selectors=["minimum original R_delta", "minimum R_star", "minimum strong R_delta"],
            tie_break="lexicographic policy name",source_sampling_not_used_for_selection=True,
            finite_confirmation=dict(eps=[1e-6,1e-4],shots=65536,seed_base=task["seed"]),
            historical_anchors=["forward","macro_reverse","check_reverse"],
            gate_resource_constraint="same directed CX multiset, measurement boundaries, slot total and each site's cumulative idle"))
    original=stim.Circuit((HISTORICAL/"results/tiny-hgp/original.stim").read_text(encoding="utf-8"))
    anchor=read_json(HISTORICAL/"results/tiny-hgp/forward/plan.json")
    side=anchor["side"]
    placement={int(q):divmod(site,side) for q,site in anchor["initial_mapping"].items()}
    plans={f"macro{m}-checks{c}":compile_round_policy(original,placement,m,c) for m in range(8) for c in range(8)}
    for order in ("forward","macro_reverse","check_reverse"):
        plans["anchor-"+order]=read_json(HISTORICAL/"results/tiny-hgp"/order/"plan.json")
    plans=match_budget(plans)
    baseline=plans["anchor-forward"]
    baseline_metrics=inherited.materialize(baseline,0)[1]
    multiset=primitive_multiset(baseline)
    rows=[]
    for name,plan in plans.items():
        target=folder/name
        if (target/"risk.json").exists():
            rows.append(read_json(target/"risk.json"));continue
        circuit,metrics=inherited.materialize(plan,.001)
        assert primitive_multiset(plan)==multiset and metrics==baseline_metrics
        commit_json(target/"plan.json",plan)
        branches,sig,d,l,total=physical_branches(circuit,.001)
        mass=np.zeros(len(sig))
        for branch in branches: mass[branch["signature"]]+=branch["coefficient"]
        faults=dict(detectors=d,observables=l,physical_coefficients=mass)
        decoder=CachedDecoder(code_for("tiny-hgp"),circuit,single_fault=faults)
        predictions=decoder.predictions(d)
        _,optimum=optimal_table(d,l,mass)
        risk={label:float(mass @ (prediction^l).any(axis=1)) for label,prediction in predictions.items()}
        save_npz(target/"single-fault.npz",**faults,**predictions)
        row=dict(order=name,R_star=optimum,R_delta=risk,metrics=metrics,matched=True,plan_sha256=digest(target/"plan.json"))
        commit_json(target/"risk.json",row);rows.append(row)
        commit_json(folder/"progress.json",dict(completed=len(rows),total=len(plans)))
    eligible=[row for row in rows if not row["order"].startswith("anchor")]
    winners={"original":min(eligible,key=lambda row:(row["R_delta"]["original"],row["order"]))["order"],
             "optimal":min(eligible,key=lambda row:(row["R_star"],row["order"]))["order"],
             "strong":min(eligible,key=lambda row:(row["R_delta"]["strong"],row["order"]))["order"]}
    commit_json(folder/"selection-before-sampling.json",dict(rows=rows,winners=winners,
        risk_optimal_and_actual_select_different_order=winners["original"]!=winners["optimal"],
        historical_forward_metrics=baseline_metrics,registration_sha256=digest(registration)))
    selected=sorted(set(winners.values())|{"anchor-forward"})
    samples=[]
    for index,name in enumerate(selected):
        circuit,_=inherited.materialize(plans[name],.001)
        faults=dict(np.load(folder/name/"single-fault.npz"))
        decoder=CachedDecoder(code_for("tiny-hgp"),circuit,single_fault=faults)
        for point,eps in enumerate((1e-6,1e-4)):
            sample_task=dict(task,order=name,eps=eps,shots=65536,seed=task["seed"]+1000*index+point)
            samples.append(decode_sample(sample_task,rescale_noise(circuit,eps/.001),decoder,folder/name/f"eps{eps}"))
    commit_json(folder/"result.json",dict(winners=winners,risks=rows,samples=samples))
    return dict(candidate_count=64,winners=winners,resource_match=baseline_metrics,confirmation_routes=len(selected))


if __name__=="__main__": finite_noise_certificates()
