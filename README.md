# Reinforcement Learning for Catalan Text Simplification with LLMs

Code for the paper **"Reinforcement Learning for improving Large Language Models' Catalan text simplification capabilities"**, presented at the CLEAR-TEXT workshop at CLIB 2026 (Sofia, Bulgaria).

- 📄 Paper: [arXiv:2609.04823](https://arxiv.org/abs/2609.04823)
- 🤗 Model: [arnauad/IberianLLM-ASSET-GRPO](https://huggingface.co/arnauad/IberianLLM-ASSET-GRPO)

## Overview

We post-train IberianLLM-7B-Instruct with **GRPO** on the ASSET dataset (original English, plus Catalan and Spanish machine translations) to improve Catalan sentence simplification. The reward combines SARI with two penalties:

- a **copy penalty** that discourages returning the source sentence unchanged;
- a **length penalty** that suppresses over-generation, a form of reward hacking where extra text inflates SARI.

```
R(X, Y, Z) = max(0, SARI/100 - P_copy) * P_len
```

Models are evaluated with SARI on the Catalan ASSET test split and on the Catalan subset of the iDEM corpus.

| Benchmark | Original | CA | ES | EN |
|-----------|---------:|---:|---:|---:|
| ASSET (ca) | 47.75 | **50.42** | 49.77 | **50.41** |
| iDEM      | 43.31 | 42.82 | 42.47 | **44.50** |

## Repository structure

```
metrics/              SARI implementation and DeBERTa model used for evaluation
scripts/              SLURM job scripts (inference, evaluation, GRPO training)
src/
├── dataset/          Dataset analysis, BLEURT filtering, merging and plotting
├── inference/        vLLM inference, ASSET translation (SalamandraTA), prompt rule search
├── evaluate/         SARI, BLEURT and REFeREE evaluation
├── prompts/          Prompts used for inference and training
└── training/         GRPO training with DeepSpeed ZeRO-3 and the reward function
```

## Usage

The scripts in `scripts/` are SLURM jobs written for the CSUC Pirineus III cluster. Run them from the `scripts/` directory after adjusting the conda environments, paths and environment variables to your setup:

```bash
sbatch inference.sh    # Generate simplifications with vLLM
sbatch evaluation.sh   # Score generations (BLEURT / SARI)
sbatch training.sh     # Multi-node GRPO training (2 nodes x 2 GPUs)
```

Training is configured through environment variables in `training.sh` (`DATASET_PATH`, `LANG`, `MODEL`, `OUT_MODEL`, `GRPO_RUNS`, `REWARDS`).

Datasets (`data/`) and outputs (`results/`) are not included in the repository. ASSET is available [here](https://github.com/facebookresearch/asset).

## Citation

```bibtex
@misc{domingo2026reinforcementlearningimprovinglarge,
      title={Reinforcement Learning for improving Large Language Models' Catalan text simplification capabilities},
      author={Arnau Ayguadé Domingo and Stefan Bott and Horacio Saggion},
      year={2026},
      eprint={2609.04823},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.04823},
}
```

## Acknowledgments

This work was carried out at Universitat Pompeu Fabra as a final degree project, using the Pirineus III supercomputer provided by CSUC. It is part of the EU Horizon Europe project iDEM (Grant Agreement No. 101132431).
