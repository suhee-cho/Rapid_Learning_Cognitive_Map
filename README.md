# Capturing Rapid Learning in an Extended Successor Representation Theory of the Cognitive Map

This repository contains the code used to generate the simulation results reported in:

**"Capturing Rapid Learning in an Extended Successor Representation Theory of the Cognitive Map"**
Suhee Cho and James L. McClelland, Department of Psychology, Stanford University

*Latest preprint version on bioRxiv (updated 25 February 2026):*
https://doi.org/10.64898/2025.12.25.696522

The repository covers three simulation paradigms:

| Paradigm | Figures generated |
|---|---|
| Linear treadmill with food reward | Main Figs. 2–3; Supp. Figs. 2–3 |
| T-maze with food and water rewards at the arm ends | Main Figs. 4–5; Supp. Fig. 6 |
| Linear maze with footshock | Main Fig. 6; Supp. Figs. 4, 5, 7 |

You can either download the example dataset (see below) or generate your own data
with new random seeds.

---

## Software requirements

Developed and tested with **Python 3.10**.

Required packages are listed in `requirements.txt`:

```
bluepyopt==1.14.16
Brian2==2.7.1
ipynb==0.5.1
matplotlib==3.9.2
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.7.2
scipy==1.14.1
seaborn==0.13.2
tqdm==4.67.3
```

Install with:

```bash
pip install -r requirements.txt
```

The offline-phase simulations build on the CA3 network model of
[Ecker et al. (2022), *eLife* 11:e71850](https://doi.org/10.7554/eLife.71850),
using their publicly available code as a starting point.

---

## Generating simulation data

To generate data from scratch, run:

```
code/generate_data.ipynb
```

Results in the paper are averaged over **10 independent runs** (*trials*), each
initialized with a different random seed. Set the variable `trial_number` to the
desired number of trials before running.

### Runtime

On a standard desktop, one online simulation lap takes roughly **20 minutes**.
Estimated runtimes per trial:

| Simulation | Laps | Approximate runtime |
|---|---|---|
| Linear treadmill (food reward) | 15 | ~5 hours |
| T-maze (food + water reward) | 10 | ~3 hours 20 minutes |
| Linear maze (footshock) | 4 | ~1 hour 20 minutes |

Generating the full 10-trial dataset for all three paradigms therefore takes on
the order of 100 hours of compute.

---

## Using the example dataset

Pre-generated data reproducing the figures in the manuscript are archived on
Zenodo:

**https://doi.org/10.5281/zenodo.21986632**

This DOI always resolves to the most recent version of the dataset.

### Download from the command line (recommended)

```bash
pip install zenodo_get
zenodo_get 10.5281/zenodo.21986632 -o data/
```

### Download manually

Download the archives from the Zenodo record above and place them in a `data/`
directory in the project root.

### Unpack

```bash
unzip data/linear_reward.zip -d data/
unzip data/Tmaze.zip -d data/
unzip data/linear_shock.zip -d data/
unzip data/factorial_control.zip -d data/
```

After unpacking, the project root should contain:

```
data/
├── linear_reward/
├── Tmaze/
├── linear_shock/
└── factorial_control/
code/
LICENSE
README.md
requirements.txt
```

---

## Reproducing the figures

Analysis notebooks are named for the figures they generate:

| Notebook | Generates |
|---|---|
| `Fig2-3_linear_reward.ipynb` | Main Figs. 2 and 3 |
| `Fig4-5_Tmaze.ipynb` | Main Figs. 4 and 5 |
| `Fig6_linear_shock.ipynb` | Main Fig. 6 |
| `FigS2_STDP_online_learning.ipynb` | Supp. Fig. 2 |
| `FigS3_global_inhibition.ipynb` | Supp. Fig. 3 |
| `FigS4_EC_strength_replay.ipynb` | Supp. Fig. 4 |
| `FigS5_Wu_replay.ipynb` | Supp. Fig. 5 |
| `FigS6_Carey_replay.ipynb` | Supp. Fig. 6 |
| `FigS7_factorial_control.ipynb` | Supp. Fig. 7 |

---

## Citation

If you use this code or dataset, please cite both the paper and the dataset:

```bibtex
@article{cho2026rapid,
  title   = {Capturing rapid learning in an extended successor representation
             theory of the cognitive map},
  author  = {Cho, Suhee and McClelland, James L.},
  year    = {2026},
  doi     = {10.64898/2025.12.25.696522}
}

@dataset{cho2026data,
  title     = {Simulation data for "Capturing rapid learning in an extended
               successor representation theory of the cognitive map"},
  author    = {Cho, Suhee and McClelland, James L.},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21986632}
}
```

If you make use of the offline-phase simulation code, please also cite
Ecker et al. (2022).

---

## License

The code in this repository is released under the MIT License (see `LICENSE`).

The offline-phase simulations are adapted from the CA3 network model of
[Ecker et al. (2022), *eLife* 11:e71850](https://doi.org/10.7554/eLife.71850),
whose code is also released under the MIT License. Files containing
adapted code are marked with a header identifying the original source and the
modifications made.

---

Hope you enjoy your time playing with the code! Questions and comments are welcome: suheecho@stanford.edu