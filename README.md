# ATL ECHO — Qwen AI Safety Public Reference Demo

**Independent research | Temporal evidence · Multi-agent source attribution · Auditable evaluation**

> **Research question:** Can internal activation evidence help distinguish the source of a controlled agent-state difference, and what controls are necessary before interpreting that difference as an early safety signal?

This repository is a **public reference implementation and evidence audit**, not the proprietary ATL ECHO algorithm. It draws its experimental framing from the author's Qwen multi-agent research files, while keeping original temporal/phase transformations and private attribution calculations undisclosed.

## Three independently interpretable modules, one entry point

| Module | What actually runs | What it does **not** establish |
|---|---|---|
| **A — Attribution evidence audit** | Audits reported aggregate 34/40 attribution and calculates a binomial reference probability | Does not rerun or independently verify the original ATL ECHO method; random-label null matching is not a false-positive rate |
| **B — Qwen open probe** | Loads `Qwen/Qwen2.5-1.5B-Instruct`, extracts last-token activations from decoder block 20 and evaluates an **open logistic-regression baseline** with scenario-held-out splits | Does not reproduce v7.7 negotiation games, predict future sabotage, or establish deception-specific detection |
| **C — Federated observation illustration** | Checks that pairwise additive masks cancel in an aggregate of synthetic vectors | Is **not** secure aggregation or a privacy guarantee |

The modules share an **evaluation question**, not a common empirical accuracy score. Results are reported separately.

## Run

Requires Python 3.10+. The full Qwen experiment downloads model weights and may require several GB of memory; a GPU is recommended. Model access and network connectivity are required for the first download.

```bash
pip install -r requirements.txt
python run_demo.py --module all
```

Run the lightweight checks without a model download:

```bash
python run_demo.py --module all --skip-qwen
```

Run only the Qwen reference experiment:

```bash
python run_demo.py --module b --scenarios 30
```

Results are written to `results/demo_results.json` (ignored by git). A failed model load is explicitly reported as **not run**, never as a successful test. Test the lightweight components with:

```bash
python -m unittest discover -s tests -v
```

### Statistical power of Module B

Module B performs forced-choice attribution over five agents, so the chance level is **20%**. With `--scenarios 5`, the outcome cannot be distinguished from chance: under random guessing, P(0 or 1 correct) ≈ 74%, and even a 4/5 result yields a two-sided 95% Wilson interval of [0.376, 0.964]. Small runs are useful as smoke tests, not as evidence.

For interpretation, at least 20–30 held-out scenarios are recommended. At an assumed 85% accuracy, the Wilson lower bound rises from 0.376 (n=5) to 0.640 (n=20) and 0.703 (n=30). Each scenario adds five forward passes, so 30 scenarios remain feasible on a single consumer GPU.

Note also that the probe operates in a high-dimensional regime: with 1,536 features and only 20 training rows at `--scenarios 5`, regularization (`C=0.1`) prevents outright failure but leaves estimates highly variable. Increasing the scenario count reduces this ratio directly.

## Scientific interpretation

**Module A.** The supplied v6.9 source contains a frozen-method validation design, with a v6.8 reference of 8/10 and a pooled 40-scenario analysis. The **34/40** figure is an *archival reported outcome*, not an outcome produced by this public demo. Under independent random five-way guessing, the corresponding binomial upper tail is approximately 1.8×10⁻¹⁸, and the two-sided 95% Wilson interval is [0.709, 0.929]. The underlying run-level CSV evidence was not provided in this package, so independent verification is not possible here. The reported binomial tail assumes independent random five-way guesses; that assumption may not describe the full experimental process.

**Module B.** The public probe is deliberately simpler than the author's v7.7 code: it uses prompt-condition differences and a held-out-scenario classifier, without recreating the original multi-round resource negotiation. Distinct instructions are present from the beginning; a positive classifier score **cannot** be interpreted as detecting a future change before behavior occurs. Because the differing instruction is itself the classification target, this module measures **prompt-condition separability**, which is a known confounder for causal source attribution rather than a control for it. Its results are therefore **not comparable** to the archival 34/40 figure. The original v7.7 research additionally explores null and benign-variant controls; those experiments are **not** reproduced here.

**Layer indexing.** Hugging Face returns `hidden_states` with length `n_layers + 1`, where index 0 is the embedding output and index *i* is the output of decoder block *i*. For Qwen2.5-1.5B-Instruct (28 blocks), decoder block 20 corresponds to `hidden_states[21]`, which is also what a forward hook on `model.model.layers[20]` captures. This repository indexes accordingly so that the reported block number matches the pilot configuration.

**Module C.** Pairwise additive masks cancel algebraically, but a deterministic demonstration without authenticated key exchange, adversarial analysis, dropout handling, and collusion controls does not establish confidentiality. In particular, an aggregate over a small population can leak information about individual contributions through differencing; cancellation correctness is not a privacy property.

## Connection to the research programme

The earlier EW/IQ investigations contributed a **validation pattern**: compare an observed measurement with randomized temporal or phase controls, and report failures as well as successes. In that signal-domain evaluation, the shuffled-phase control did **not** separate the observed score (empirical p = 0.425), and the failure was traced to dispersion inflation rather than to absence of temporal structure. The same concern applies to separation margins in the activation-space work, where the highest observed margin occurred in a null episode. The Qwen experiments explore whether internal model activations offer source-attribution evidence. Later architectural research asks how such evidence can remain traceable and auditable. These are methodological connections; the same physical mechanism is **not** asserted across domains.

## Publication and research

- [ATL ECHO — AI Safety Research](https://github.com/gizemakinciisten/ATL-ECHO-AI-Safety-Research)
- [ATL ECHO — Anti-Time Layer Echo](https://github.com/gizemakinciisten/ATL-ECHO-Anti-Time-Layer-Echo)
- [ATL ECHO — Public Evidence Demo](https://github.com/gizemakinciisten/ATL-ECHO-Temporal-Evidence-Demo)
- [G. Akıncı Bertan (2025), *Cumhuriyet Science Journal*](https://doi.org/10.17776/csj.1574774)
- [ORCID](https://orcid.org/0009-0009-8555-9690)

## License

The demonstration code in this repository is released under the **Apache License 2.0** — see [`LICENSE`](LICENSE).

This license applies **only to the public demo files distributed here**: the entry point, the three module implementations, the tests, and the accompanying documentation. It grants no license to proprietary ATL ECHO implementations, which are not distributed in this repository. Third-party components remain under their own licenses; Qwen model weights are downloaded separately from Hugging Face and are governed by their published license terms.

## Disclosure and limitations

This code contains **no proprietary ATL ECHO phase/amplitude operators, frozen Loschmidt scoring implementation, private calibration, or raw experimental activation records**. All reported original outcomes are distinguished from fresh public runs. No security, privacy, or model-generalization guarantee is made.
