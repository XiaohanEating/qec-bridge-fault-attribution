# Experimental protocol

The objective is to compare gate orders with the same ideal measurement task and resources, and to attribute their single-fault risk to physical ambiguity and decoder decision loss. The artifact does not change the algorithms, scientific weights, seeds, or shot counts used for the manuscript.

## Circuit and decoder conditions

- The primary repetition-check HGP construction is a distance-three unrotated surface code; its saved indexing imposes a remote layout. The second instance uses H3/H4 repetition checks. Both use an initial noisy extraction plus three noisy repetitions, with ideal final data readout.
- Nonlocal BB-72 and BB-108 circuits use QUITS natural checkerboard placements with periodic and nonlocal interactions routed on open grids. All logical-Z outputs are retained jointly.
- SD6 assigns independent probability epsilon to each noisy gate, idle, preparation, and readout location. Physical Pauli weights are 1/3 for each one-qubit depolarizing branch, 1/15 for each two-qubit branch, and 1 for preparation/readout flips.
- Original and strong sliding-window BP-OSD use width/advance 1/1 and 4/4, BP limits 2 and 50, and combination-sweep OSD orders 0 and 2. Serial product-sum BP, circuit matrices, and logical mappings are retained. Scan priors are fixed at 1e-3; matched-prior controls are separate.
- The optimal single-fault lookup selects the greatest physical mass per nonzero detector record, predicts zero on the noiseless record, and uses strong BP-OSD outside its table. Its single-fault coefficient and finite-noise failure rate are distinct quantities.

## Statistical comparisons

Zero- and one-fault contributions are exact. Conditional multi-fault sampling chooses distinct physical locations without replacement, draws physical Pauli branches, and XORs their Clifford responses. Every sampled fault-count layer has 32,768 shots. Unsampled probability is retained as an explicit tail bound.

For the two historical instances, three decoder rules, two candidate-minus-forward orders, and five rates from 1e-6 to 1e-4, the 60 differences use simultaneous 95% intervals. Error probability is divided across both circuits and their sampled layers. Direct circuit comparisons use independent seeds; decoder comparisons reuse the same physical samples.

The 4096-shot parameter exploration is a paired reanalysis, not an independent confirmation. Single-factor comparisons use an exact McNemar test with a 96-test Holm family; fixed/matched-prior comparisons use a separate 54-test family. The historical 8192-shot paired decoder experiment is labeled separately.

The 64-order search keeps the initial noisy block forward and selects by original, strong, or optimal single-fault risk. Selected orders receive fresh 65,536-shot confirmations. Their 12 simultaneous differences are distinct from the historical three-order comparisons. BB uses eight rates from 1e-9 to 1e-4, 32,768 shots per circuit/rate, with 96 simultaneous ordering differences.

## Hardware controls and negative outcomes

Native extraction, arbitrary-state-restoring bridges, and prepared-auxiliary Zhao routing have different hardware assumptions. Directed-edge CX counts, slots, per-site idle exposure, auxiliary preparation, and long-range couplers remain explicit. The local dynamic-route exchange and its lack of detected improvement are preserved. High BB failure rates are included as measured.

Original registered sources and execution-equivalence evidence are retained. Runtime accelerators were checked against the original decoder's error vectors, soft information, and logical predictions; concurrent elapsed time is not reported as a standalone deployment performance measurement.

