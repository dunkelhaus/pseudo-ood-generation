# pseudo-ood-generation

An implementation of the Pseudo OOD Generation model (POG), detailed in
<https://arxiv.org/pdf/1909.03862.pdf>.

## Supported CUDA API

Use `pseudo_ood_generation.builders.build_model` and `build_trainer` for the
word-level POG and AEPOG models. The package uses the nested `wgan_gp` package
for latent generator and critic updates. Its supported imports do not require
AllenNLP, BERT, or image-processing dependencies.

`pseudo_ood_generation/components/ae/builders.py` preserves an earlier
AllenNLP/BERT implementation. Its constructor calls no longer match the current
autoencoder, so it is retained as historical source rather than a supported
entrypoint. The main repository's `oos_detect.reproduce` commands provide the
artifact-driven training and evaluation path.
