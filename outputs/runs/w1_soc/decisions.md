# W1 SOC Decisions

## 2026-05-07

- Use strict NASA SOC reconstruction for reporting runs: each discharge cycle is Coulomb-counted independently, starts at SOC=1, and calibrates the terminal SOC from capacity consistency plus voltage cutoff.
- Use B0005/B0006/B0007 as NASA internal SOC training cells and B0018 as the holdout cell when checking whether ARC-to-PCoE domain shift is the blocker.
- Keep `outputs/soc_finetuned.h5` as the current best available W1 SOC artifact, copied from the B0005/B0006/B0007 -> B0018 run. Current B0018 holdout MAE is 3.48%, so the W1 `<1.5%` acceptance item remains open.
- Do not use full LSTM unfreeze as the default SOC strategy yet. Full unfreeze with LR=1e-5 and LR=1e-6 lowered train MAE but worsened B0018 holdout to 5.81% and 5.27%, respectively, which indicates cell-domain overfitting rather than under-adaptation from frozen early LSTM layers.
- Figures produced from this model are acceptable as preliminary pipeline/trajectory visuals, not as final SOC accuracy evidence for the paper.
