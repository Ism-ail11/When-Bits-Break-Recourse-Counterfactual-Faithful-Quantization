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
- Constraint tightness, distribution shift, subgroup reporting, teacher quality/noise, budget curves, runtime, robust-solver, and theory diagnostics.


## Metric conventions

- **VD:** target failure under the quantized model among examples for which FP recourse was feasible.
- **CRG:** relative cost change on examples for which both FP and quantized recourse were feasible.
- **Infeasible recourse:** reported separately through FP/Q feasible rates rather than hidden inside an arbitrary cost penalty.

