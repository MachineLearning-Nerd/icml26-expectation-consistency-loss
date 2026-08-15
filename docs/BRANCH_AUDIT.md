# Branch audit and naming policy

The original repository used `master` plus twelve `orx/*` work branches. Every
public tip is preserved under a purpose-based final name. `main` is the only
publication surface; the other branches are inspectable experiment or release
histories.

| Original ref | Tip | Final ref | Purpose |
| --- | --- | --- | --- |
| `master` | `0566fc3` | `main` | Integrated status, illustrated report, evidence bundle, and reader-facing surface. |
| `orx/frozen-cumulative-baseline` | `c0d9f66` | `experiment/frozen-cumulative-baseline` | Freeze the accepted baseline environment and campaign runner. |
| `orx/claim-2-corrected-finite-sample-theorem` | `c2862f8` | `audit/claim-2-corrected-finite-sample` | Corrected hard/soft finite-sample proof route. |
| `orx/claim-2-route-3-soft-bin-concentration-proof` | `647e2ed` | `audit/claim-2-soft-bin-concentration` | Final soft-bin concentration proof and independent diagnostics. |
| `orx/claim-2-synthesis-and-claim-4-real-mnist-soft-bi` | `71002c3` | `audit/claim-2-and-4-real-mnist` | Full-MNIST differentiable ECL and sample-complexity evidence. |
| `orx/claim-2-universal-falsification-stress` | `66d09da` | `audit/claim-2-falsification-stress` | Probability-domain stress search for a valid C2 counterexample. |
| `orx/claim-5-route-1-faithful-lenet5-svhn-reconstruct` | `4ad7404` | `audit/claim-5-lenet-svhn` | Literal/full-domain LeNet-SVHN reconstruction route. |
| `orx/claim-5-route-2-predecessor-post-hoc-lenet5` | `e4f89e9` | `audit/claim-5-predecessor-posthoc` | Public-predecessor post-hoc LeNet route. |
| `orx/claim-5-route-3-stabilized-appendix-j-lenet5` | `02a3f60` | `audit/claim-5-stabilized-appendix` | Stabilized Appendix-J LeNet route. |
| `orx/claim-5-route-4-mandatory-falsification-audit` | `f562174` | `audit/claim-5-falsification` | Assumption-complete attempt to falsify the historical Table 2 aggregate. |
| `orx/claim-6-route-1-full-simulated-figure-2` | `4c1dd91` | `audit/claim-6-simulation` | Five-seed paper-text simulation for Figure 2. |
| `orx/claim-6-route-2-primary-source-table-1-audit` | `cb25d98` | `audit/claim-6-table1-falsification` | Exact Table 1 conjunction falsification through C3. |
| `orx/release-candidate-evidence-report-and-logbook` | `27f6b26` | `release/candidate-evidence` | Release-candidate report, notebook, and artifact assembly. |

All final branches are normalized to
`MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>` before
publication. The original names are retained in this audit table as provenance,
not as live branch references.

## Reading order

1. Read `main/README.md` for the claim matrix and limits.
2. Follow `docs/CLAIM_EVIDENCE.md` to the producer and committed artifact for
   each claim.
3. Use the final branch names above to inspect an individual route. A branch
   tip is historical evidence of that route, not an automatic endorsement of
   its strongest interim wording.
