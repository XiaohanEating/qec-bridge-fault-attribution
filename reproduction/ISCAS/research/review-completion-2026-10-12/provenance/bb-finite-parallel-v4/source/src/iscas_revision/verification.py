"""Physical cross-checks required before conditional sampling is accepted."""
import time
import numpy as np
import stim
from .artifacts import HISTORICAL, WORK, read_json, commit_json, digest
from .stratification import PhysicalResponses, NOISE, bind_branch_locations
from .decoding import CachedDecoder
from support import code_for


def run():
    records=[]
    for instance in ("tiny-hgp", "holdout-hgp-3x4"):
        folder=HISTORICAL/"results"/instance/"forward"
        prior=HISTORICAL/"provenance/decoder-priors"/instance/"forward.stim"
        circuit=stim.Circuit(prior.read_text(encoding="utf-8"))
        branches=bind_branch_locations(circuit,read_json(folder/"branches.json"))
        faults=np.load(folder/"single-fault.npz")
        model=PhysicalResponses(circuit,branches,faults)
        rng=np.random.default_rng(981012)
        injection_checks=[]
        for trial in range(8):
            selected=rng.choice(len(branches["branches"]),size=2,replace=False)
            items=[branches["branches"][int(index)] for index in selected]
            while (items[0]["instruction_offset"],items[0]["target_start"]) == (items[1]["instruction_offset"],items[1]["target_start"]):
                selected=rng.choice(len(branches["branches"]),size=2,replace=False)
                items=[branches["branches"][int(index)] for index in selected]
            injected=stim.Circuit()
            for offset,inst in enumerate(circuit.flattened()):
                for item in items:
                    if item["instruction_offset"] == offset:
                        for q,pauli in item["paulis"]: injected.append(pauli+"_ERROR",[q],1.0)
                if inst.name not in NOISE: injected.append(inst)
            d,l=injected.compile_detector_sampler(seed=981012+trial).sample(16,separate_observables=True)
            i,j=[item["signature"] for item in items]
            assert np.all(d == (faults["detectors"][i]^faults["detectors"][j]))
            assert np.all(l == (faults["observables"][i]^faults["observables"][j]))
            injection_checks.append(dict(branch_indices=selected.tolist(),shots=16,equal=True))
        d,l,sites,sig=model.sample(7,4096,981019)
        assert np.all(np.diff(np.sort(sites,axis=1),axis=1)>0)
        assert np.array_equal(np.bitwise_xor.reduce(model.responses[sig],axis=1),np.concatenate([d,l],axis=1))
        decoder=CachedDecoder(code_for(instance),circuit,single_fault=faults)
        rows=np.linspace(0,len(faults["detectors"])-1,32,dtype=int)
        predictions=decoder.predictions(faults["detectors"][rows])
        assert all(np.array_equal(predictions[label],faults[label][rows]) for label in ("original","strong"))
        records.append(dict(instance=instance,prior_sha256=digest(prior),noise_locations=model.n,
            conditional_sites_distinct=True, response_xor_equal=True, deterministic_two_fault_injections=injection_checks,
            historical_decoder_predictions_equal=True))
        print({"verification":instance,"passed":True},flush=True)
    commit_json(WORK/"provenance/physical-verification.json",dict(passed=True,records=records))


if __name__ == "__main__": run()
