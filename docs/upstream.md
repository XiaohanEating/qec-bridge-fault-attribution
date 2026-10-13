# Upstream sources

The release imports installed QUITS, Stim, ldpc, PyMatching, and the scientific Python packages. Their own licenses apply. Exact versions are recorded in `environment/versions.json`.

Official control implementations are fetched separately so that their source and terms remain attached to the original repositories:

| Control | Source | Commit |
|---|---|---|
| Hirai et al., No more hooks in the surface code | https://github.com/yugahirai/num_exp_no_more_hooks_in_the_surface_code | `32cef750e67548838513191eaed8b58673fe00ec` |
| Zhao et al., A simple universal routing strategy | https://github.com/Guangqi-Phys/Routing-Circuit | `295adf67633f223e1cc543d972cd34012c10a4e8` |

`qec-bridge fetch-upstream` checks out exactly those commits. Saved control samples and the artifact's own adaptations are included in the data release. Hirai's original Stim circuits and PyMatching controls retain the author's uniform depolarizing model. Zhao's author-noise and common-SD6 controls are labeled separately, because preparation/readout axes and idle accounting differ. Native surface-code controls use Stim's generated unrotated memory circuits.

The paper's bridges restore arbitrary states of intermediate qubits. Zhao uses prepared auxiliaries and long-range couplers. These hardware assumptions must be retained when comparing resources or reliability.

