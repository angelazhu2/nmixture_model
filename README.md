# Building MCMC samplers for N-Mixture Models
By Angela Zhu and Brayden Edwards 

This project implements and compares five MCMC methods for fitting an N-mixture model — a hierarchical model used in ecology to estimate wildlife abundance when animals are imperfectly detected.

## Model

The N-mixture model assumes:

- True abundance at site $i$: $N_i \sim \text{Poisson}(\lambda)$
- Observed count at site $i$, visit $t$: $C_{it} \sim \text{Binomial}(N_i, p)$

The goal is to recover $\lambda$ (expected abundance), $p$ (detection probability), and per-site true abundances $N_i$ from repeated count surveys $C$.


## Usage
The `data\` folder contains both our real world data and our results. 

#### Run Methods
Navigate to the `src` folder and run `python main.py` this is where we set up parameters and run methods. 
Results are saved automatically to `data/results/`. This folder contains our results from multiple runs. 

Method code is all stored within files names `method_x.py` while the helper functions like calcuating joint probabilities and acceptance probabiltiies are within `utils.py`.

#### Simulation Data
The code for this can be found in `utils.py` within the helper function `forward_pass()`. For the code to build simulation data that is within `build_dataset.py`


### Parameters

| Parameter | Description |
|-----------|-------------|
| `sites`   | Number of survey sites |
| `T`       | Number of repeat visits per site |
| `lam`     | True Poisson rate for abundance |
| `p`       | True detection probability |
| `S`       | Upper bound on lambda (used as proposal range) |
| `EPOCHS`  | Number of MCMC iterations |

## Methods

All methods use Metropolis-Hastings MCMC. They differ in how proposals are generated for $\lambda$, $p$, and $N$.

| Method | Description |
|--------|-------------|
| **Method 1** | Independent uniform proposals for all parameters ($\lambda$, $p$, $N$). Simple but low acceptance rate. |
| **Method 2** | Uniform proposals for $\lambda$ and $p$; $N$ proposed from Poisson($\lambda$). Uses Hastings correction for asymmetric proposals. |
| **Method 3** | Random walk proposals: $\lambda$ from Normal($\lambda$, 1), $p$ from Normal($p$, 0.1), $N_i$ from Poisson($N_i$) truncated at $C_{\max,i}$. |
| **Method 4** | Component-wise (Metropolis-within-Gibbs): updates $\lambda$, $p$, and each $N_i$ one at a time with up to 5 attempts per component per iteration. |
| **Method 5** | Random walk with truncated normal proposals for $\lambda$ and $p$ (corrects for boundary effects), and truncated Poisson for $N_i$. Fully corrects Hastings ratio. |

Evaluated on simulated data with default parameters: `lam=5`, `p=0.25`, `T=6`, `R=20`.



