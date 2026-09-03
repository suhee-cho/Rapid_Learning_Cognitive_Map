"""
stat_functions.py  [NEW]
========================
Effect-size and confidence-interval helpers used by the analysis notebooks to
report the statistics quoted in the manuscript.

Functions
---------
  cohens_d_paired — Cohen's d for paired samples (mean difference / SD of the
                     differences).
  cohens_d_ind    — Cohen's d for two independent samples (pooled SD).
  cohens_d_1samp  — Cohen's d for one sample against a reference value mu0.
  ci_paired       — two-sided (1 - alpha) t CI for the mean paired difference.
  ci_ind          — two-sided (1 - alpha) t CI for the difference between two
                     independent means.
  ci_1samp        — two-sided (1 - alpha) t CI for a single-sample mean.

All functions in this file are new (written for this project).
"""

import numpy as np
from scipy import stats

def cohens_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

def cohens_d_ind(x, y):
    x, y = np.array(x), np.array(y)
    n1, n2 = len(x), len(y)
    pooled_std = np.sqrt(((n1-1)*np.var(x, ddof=1) + (n2-1)*np.var(y, ddof=1)) / (n1+n2-2))
    return (np.mean(x) - np.mean(y)) / pooled_std

def cohens_d_1samp(x, mu0):
    x = np.array(x)
    return (np.mean(x) - mu0) / np.std(x, ddof=1)

def ci_paired(x, y, alpha=0.05):
    diff = np.array(x) - np.array(y)
    n = len(diff)
    se = stats.sem(diff)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    return np.mean(diff) - t_crit*se, np.mean(diff) + t_crit*se

def ci_ind(x, y, alpha=0.05):
    x, y = np.array(x), np.array(y)
    se = np.sqrt(stats.sem(x)**2 + stats.sem(y)**2)
    t_crit = stats.t.ppf(1 - alpha/2, df=len(x)+len(y)-2)
    return (np.mean(x)-np.mean(y)) - t_crit*se, (np.mean(x)-np.mean(y)) + t_crit*se

def ci_1samp(x, alpha=0.05):
    x = np.array(x)
    n = len(x)
    se = stats.sem(x)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    return np.mean(x) - t_crit*se, np.mean(x) + t_crit*se