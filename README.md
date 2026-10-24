# Classifying word-sense definitions into supersenses with LLMs

Lab 1 of the M2 course NLP for Society (Marie Candito), Computational Linguistics.
Botakoz Kadyrbay and Alena Muravyeva.

We classify French Wiktionary noun senses (lemma and definition) into 25 supersenses, and through a fixed mapping
into 12 hypersenses, and compare:

1. in-context learning (ICL) with a 4-bit instruction-tuned LLM (`Qwen2.5-7B-Instruct`);
2. QLoRA fine-tuning of the same model on the training set (label generation vs classification head);
3. full fine-tuning of a French encoder (CamemBERT), the optional baseline of the lab.

Data: [Angleraud et al., 2025](https://aclanthology.org/2025.coling-main.356/), taxonomy from
[Barque et al., 2020](https://aclanthology.org/2020.lrec-1.724/). The `protect-frsemcor` senses are removed.
All choices are tuned on `rand-dev`; final results are computed once on `rand-test` and `freq-test`.

## Results

Final configuration of each method, chosen on `rand-dev` and run once on the test sets
(supersense macro-F1 / accuracy; all runs in [`results/metrics.csv`](results/metrics.csv)):

| Method | Final configuration | rand-test | freq-test |
|---|---|---|---|
| ICL | English prompt, 5 similar demonstrations, no lemma, label scoring | 0.583 / 0.717 | 0.519 / 0.523 |
| QLoRA | adapters on all linear layers, label generation, with lemma | 0.719 / 0.820 | 0.675 / 0.702 |
| CamemBERT | `camembert-large`, with lemma | 0.705 / 0.807 | 0.639 / 0.682 |

QLoRA and CamemBERT are better than ICL in 100 % of paired bootstrap resamples on both test sets. QLoRA and
CamemBERT cannot be distinguished on `rand-test`; QLoRA is better on `freq-test`.

## Notebooks

| Notebook | Content | Runtime | Colab |
|---|---|---|---|
| `00_data_exploration` | label distributions per split, properties of the inputs | CPU | [open](https://colab.research.google.com/github/Alhassliebe/LAB1_classification_with_llms/blob/main/notebooks/00_data_exploration.ipynb) |
| `01_icl` | prompt language, number and choice of demonstrations, lemma, generation vs label scoring | GPU | [open](https://colab.research.google.com/github/Alhassliebe/LAB1_classification_with_llms/blob/main/notebooks/01_icl.ipynb) |
| `01b_icl_error_analysis` | errors of the best ICL configuration | CPU | [open](https://colab.research.google.com/github/Alhassliebe/LAB1_classification_with_llms/blob/main/notebooks/01b_icl_error_analysis.ipynb) |
| `02_lora` | LoRA modules, generation vs label scoring, generation vs classification head, lemma | GPU | [open](https://colab.research.google.com/github/Alhassliebe/LAB1_classification_with_llms/blob/main/notebooks/02_lora.ipynb) |
| `03_camembert` | lemma, `camembert-base` vs `camembert-large` | GPU | [open](https://colab.research.google.com/github/Alhassliebe/LAB1_classification_with_llms/blob/main/notebooks/03_camembert.ipynb) |
| `04_error_analysis` | comparison of ICL and QLoRA: shared errors, per-class F1, input properties | CPU | [open](https://colab.research.google.com/github/Alhassliebe/LAB1_classification_with_llms/blob/main/notebooks/04_error_analysis.ipynb) |

Each notebook starts with a setup cell that clones this repository on Colab or Kaggle, so that `src/` is importable;
the dataset is downloaded on first use. GPU notebooks were run on a T4. Every run is saved to
`results/predictions/` and `results/metrics.csv`; when a prediction file already exists, the notebook reuses it
instead of running the model again.

## Repository layout

```
src/data.py              loading, splits, supersense to hypersense mapping, input formatting
src/evaluate.py          accuracy, macro-F1, paired bootstrap, prediction files, experiment log
src/prompts.py           ICL prompts, choice of demonstrations, parsing of the answers
src/label_scoring.py     log-probability of each label after the prompt
notebooks/               experiments, one notebook per method, plus data exploration and error analysis
docs/                    notes on the choice of the model, the prompting literature and the LoRA setup
results/metrics.csv      one row per run; every number in the report comes from this file
results/predictions/     one CSV of predictions per run
results/figures/         figures of the report
```

## Running locally

```bash
pip install -r requirements.txt
jupyter lab notebooks/00_data_exploration.ipynb
```
