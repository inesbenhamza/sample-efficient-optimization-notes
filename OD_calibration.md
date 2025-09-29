# OD Calibration with SUMO and Bayesian Optimization


## 🎯 Problem Statement
In transportation networks, we often **do not know the true Origin–Destination (OD) matrix** (i.e., how many vehicles travel from each origin zone to each destination zone).  
What we do have are **ground-truth edge counts** from detectors (e.g., how many vehicles passed through certain roads during a given time slot).  

The goal of **OD calibration** is to **estimate the OD matrix** that best reproduces the observed traffic counts. In essence, the problem is to find the OD matrix 𝑥 such that, when simulated in SUMO, it reproduces the observed edge counts as closely as possible.

## Formulation
We define an optimization problem:

\[
x^* = \arg\min_x \; \mathrm{NRMSE}\big(S(x), y_{\text{gt}}\big)
\]

Where:
- \(x\) = OD demands (number of vehicles per O–D pair).  
- \(S(x)\) = simulated edge counts produced by **SUMO** when running the network with OD matrix \(x\).  
- \(y_{\text{gt}}\) = measured edge counts from detectors.  
- NRMSE = Normalized Root Mean Squared Error, used as the error metric.

---

## Why Simulation?
There is no simple formula linking OD demand to edge counts.  
Traffic flows depend on:
- route choices,  
- congestion,  
- network interactions.  

Therefore, the only way to evaluate a candidate OD matrix \(x\) is to **run it in SUMO** and measure the resulting edge counts.

## Optimization with Bayesian Optimization
1. **Initialization**: sample candidate OD matrices (e.g., via Sobol sampling).  
2. **Simulation**: run each OD in SUMO → get simulated edge counts.  
3. **Loss computation**: compare simulated vs observed counts using NRMSE.  
4. **BO loop**:
   - Train a surrogate model (Gaussian Process) on the evaluated \((x, f(x))\) pairs.  
   - Use an acquisition function (e.g., Expected Improvement) to propose a new OD matrix.  
   - Repeat simulation + loss computation.  

This process iteratively refines the estimate of the OD matrix that best matches reality.

---

## In One Sentence
The point of the simulation is **to iteratively search for the OD matrix that minimizes the difference between simulated edge counts and observed ground-truth counts**, thereby recovering the most plausible “true” OD flows.

