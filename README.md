# Capturing Rapid Learning in an Extended Successor Representation Theory of the Cognitive Map

This repository contains the code used to generate the simulation results included in the manuscript:

**"Capturing Rapid Learning in an Extended Successor Representation Theory of the Cognitive Map."**

*Latest version on bioRxiv (last updated on Feb 25th, 2026):*
https://doi.org/10.64898/2025.12.25.696522

The repository includes three different simulation paradigms:

- **Linear treadmill with food reward** (used to generate main figures 2 and 3)
- **T-maze with food and water rewards at each arm** (used to generate main figures 4 and 5)
- **Linear treadmill with footshock** (used to generate main figure 6)

You can either use the example datasets provided in this repository or generate your own data using new random seeds.

---
# Software Requirements

The code was developed and tested using **Python 3.8.10**.

The following Python libraries are required:
```
bluepyopt==1.14.16
Brian2==2.7.1
ipynb==0.5.1
matplotlib==3.9.2
numpy==1.26.4
pandas==3.0.1
scikit-learn==1.7.2
scipy==1.14.1
sns==0.13.2
tqdm==4.67.3
```

You can install the required dependencies using:
```
pip install -r requirements.txt
```

---

# Generating Simulation Data

To generate your own data, run the notebook:

`code/generate_data.ipynb`

Note that the results reported in the paper are based on averages over **10 independent runs** (referred to as *trials*), each initialized with a different random seed.

Before running the simulations, set the variable:

`trial_number`

to the desired number of trials.

### Runtime Expectations

On a standard computer, generating data for **one online simulation lap** takes approximately **20 minutes**.

Estimated runtimes for a single trial:

| Simulation | Laps | Approximate Runtime |
|-------------|------|---------------------|
| Linear treadmill (food reward) | 15 laps | ~5 hours |
| T-maze (food + water reward) | 10 laps | ~3 hours 20 minutes |
| Linear treadmill (footshock) | 4 laps | ~1 hour 20 minutes |

---

# Using Example Data

If you prefer to use pre-generated data to reproduce the results from the manuscript, you can download the example data from Google Drive.

### Download the data by running a command (recommended)

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1-C93zsbEiat-fKGBOSCEpvC3OcdtIM45
```

### Download the data manually
**Download link:**  
[Download dataset](https://drive.google.com/drive/folders/1-C93zsbEiat-fKGBOSCEpvC3OcdtIM45?usp=sharing)

### Instructions
1. Download 'data' folder from GooglE Drive.
2. Move the downloaded folder into the project root directory.
3. Unzip the files by running:

```bash
unzip data/linear_reward.zip
unzip data/Tmaze.zip
unzip data/linear_shock.zip
---

# Data Analysis

We provide separate Jupyter notebooks for analyzing the outputs of different simulation scenarios: 

<mark style="background-color: lightyellow">analyze_linear_reward.ipynb</mark> - Generate results panels of main figures 2 and 3.

<mark style="background-color: lightyellow">analyze_Tmaze.ipynb</mark> - Generate results panels of main figures 4 and 5.

<mark style="background-color: lightyellow">analyze_linear_shock.ipynb</mark> - Generate results panels of main figure 6.

### Using Example Data

If you want to analyze the example data included in the repository, set:
`use_example = True`

in the first code block.

### Using Your Own Generated Data

If you generated new simulation data, set:
`trial_number`

to the number of trials (random seeds) used during data generation.

---

Hope you enjoy your time playing with the code! If you have any questions, please contact me through email: suheecho@stanford.edu.