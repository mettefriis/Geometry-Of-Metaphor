# Geometry-Of-Metaphor
Code for "The Geometry of Metaphor: Probing and Steering Metaphor Representations in Large Language models" (BlackboxNLP 2026). Paper: https://arxiv.org/pdf/2610.31520

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login
```

Put the Gutiérrez et al. (2016) dataset at `data/gut-2016-new.csv`.

## Run

Run the stages in this order, submitting the jobs from inside each folder:

```bash
cd 01_domain_general_das && for j in jobs/*.job; do sbatch $j; done
cd ../03_steering        && for j in jobs/*.job; do sbatch $j; done
cd ../04_generation      && for j in jobs/*.job; do sbatch $j; done
cd ../05_judge           && for j in jobs/*.job; do sbatch $j; done
```

Wait for each stage to finish before starting the next. Replace the `module` and `source` lines in the jobs with your own environment.
