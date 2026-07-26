# Counterfactual-Faithful Quantization (CFQ)

A clean PyTorch reference implementation for the paper **“When Bits Break Recourse: Counterfactual-Faithful Quantization.”** The repository implements the method, metrics, actionable projections, quantizers, mixed-precision allocation, post-training variant, baselines, and experiment families described in the main text and appendix.


## Implemented components

- Validity Drop (VD), Counterfactual Recourse Gap (CRG), direction similarity, action overlap, feasible recourse rate, and target-margin diagnostics.
- Action sets with immutable features, box constraints, top-k sparsity, one-hot projection, ordinal projection, and straight-through gradients.
- Training and evaluation PGD recourse solvers, a binary linear L2 sanity-check solver, and a robust solver over quantization variants.
- LSQ-style learned weight steps, PACT-style activation clipping, hard/soft Gumbel mixed precision, bit-specific quantizer parameters, and budget penalties.
- CFQ-QAT, uniform QAT, accuracy-centric mixed precision, CF-PTQ, counterfactual sensitivity allocation, R-Margin, R-Consistency, pruning+quantization, and distillation+quantization.
- Adult, German Credit, COMPAS, Bank Marketing, Default of Credit Card Clients, and synthetic tabular loaders.
- MNIST and Fashion-MNIST latent-recourse experiments with a convolutional autoencoder.
- A strict `SemanticEditor` interface for CelebA instead of silently substituting unrealistic pixel perturbations.
- Constraint tightness, distribution shift, subgroup reporting, teacher quality/noise, budget curves, runtime, robust-solver, and theory diagnostics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The tabular public datasets are downloaded on first use. MNIST and Fashion-MNIST are downloaded through `torchvision`. COMPAS is cached from the ProPublica analysis repository.

## Verify the installation

```bash
cfq smoke --output-dir results/smoke
pytest
```

The smoke command trains a tiny FP model, performs CFQ-QAT, generates constrained recourse, and writes metrics/checkpoints.

## Run one tabular experiment

```bash
cfq tabular \
  --config configs/base.yaml \
  --dataset adult \
  --method cfq \
  --output-dir results/adult/cfq
```

Supported method names include:

```text
fp32, lsq, pact, mixedprec, cfq, cfq_uniform, cfq_match,
ptq4, ptq8, mixedptq, cfptq, cfptq_sensitivity,
r_margin, r_consistency, prune_quant, kd_quant
```

## Reproduce experiment families

```bash
python scripts/run_main.py
python scripts/run_ablation.py
python scripts/run_budget_curves.py
python scripts/run_backbones_costs.py
python scripts/run_more_baselines.py
python scripts/run_cfptq.py
python scripts/run_teacher_quality.py
python scripts/run_runtime.py
python scripts/run_stress_tests.py
python scripts/run_theory_diagnostics.py
python scripts/run_non_tabular.py
```

Run the compact validation suite:

```bash
python scripts/run_all.py --fast
```

Run the complete matrix:

```bash
python scripts/run_all.py
```

The complete matrix is computationally expensive because evaluation recomputes high-accuracy recourse for both FP and quantized models.

## Script-to-paper map

See [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md). Every main-text and appendix experiment family is mapped to an executable script and output file.

## Output format

Each run directory contains:

```text
config.json
metrics.json
fp_model.pt
quantized_model.pt
recourse.pt
```

Suite scripts additionally write CSV summaries. Metrics record the conditioning convention used for VD and CRG.

## Metric conventions

- **VD:** target failure under the quantized model among examples for which FP recourse was feasible.
- **CRG:** relative cost change on examples for which both FP and quantized recourse were feasible.
- **Infeasible recourse:** reported separately through FP/Q feasible rates rather than hidden inside an arbitrary cost penalty.

These conventions are configurable in code but fixed across compared methods in every supplied runner.




## Citation

@misc{yahyati2026whenbitsbreakrecourse, title = {When Bits Break Recourse: Counterfactual-Faithful Quantization}, author = {Yahyati, Chaymae and Lamaakal, Ismail and El Makkaoui, Khalid and Ouahbi, Ibrahim}, year = {2026}, eprint = {2605.17160}, archivePrefix = {arXiv}, primaryClass = {cs.LG}, url = {https://arxiv.org/abs/2605.17160} }
