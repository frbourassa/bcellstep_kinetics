# Population kinetics analysis of B cell activation by VLPs

## Calcium traces processing for Fig. S2

For each recording (defined by VLP concentration and replicate), we first filtered out spurious calcium fluorescence traces, excluding traces lasting fewer than 50 frames or beginning more than 30 s after the start of the recording. We retained only replicates containing at least 15 cells to ensure meaningful fits could be performed.

To distinguish inactive from active B cells, we defined a calcium fluorescence intensity threshold equal to a factor $f$ times the background fluorescence; the background was determined by finding the minimum fluorescence in each trace and taking the median over all traces. We used the same factor $f$ across recordings. We tested a range of $f$ values between 1.1 and 2.5 and settled on a default value $f = 1.8$ (Fig. S2G-I).

To determine the activation time $t^{\star}$ of each B cell (i.e., the time at which its calcium trace first crossed the threshold), we performed linear interpolation between the last time point below threshold and the first time point above it. Traces already above threshold at the start of imaging were assigned $t^{\star} = 0$. Cells that never crossed the threshold were labeled inactive for the entire recording. Representative calcium fluorescence traces and the thresholding procedure are illustrated in Fig. S2D.


## Empiric cumulative distribution function (CDF) normalization

In principle, tracks that do not reach activation during a recording carry information about the activation time distribution; however, in practice, some of those tracks come from anergic, dead, or out-of-focus B cells, or from otherwise spurious tracks. To estimate the fraction of such inactive tracks to reject, we reasoned that at high (10 pM) particle concentrations, all B cells should have encountered enough particles to activate. We therefore used the fraction of inactive tracks in those datasets as an estimate of the fraction of tracks to reject, and rejected at most this fraction of inactive tracks from all other datasets. This still left, for instance, $\sim 25\ \%$ inactive tracks at the end of recordings for 0.01 pM concentration, where the particle contact rate is low enough that several cells had not yet been activated.


## Mathematical modeling of the distribution of B cell activation times

We treated B cell activation as a stochastic process governed by the waiting time for a B cell to encounter one or more randomly diffusing VLPs and subsequently reach the calcium activation threshold. The activation time $t^{\star}$ thus corresponds to the first-passage time (FPT) for the B cell to reach the "active" state. The CDF of the FPT, $F_{t^{\star}}(t) = P(t^{\star} < t)$, gives the probability that activation has occurred at or before time $t$. 

### Effective activation rate

First, we extracted an "observed" activation rate for each experimental replicate using a one-step activation model, in which a B cell transitions from inactive to active with an effective rate $k_\mathrm{obs}$. In this simplest possible model, the FPT follows an exponential distribution:

$$f_{t^{\star}}(t) = k e^{-kt} \qquad \text{and} \qquad F_{t^{\star}}(t) = 1 - e^{-kt} \ \ .$$

We fitted this distribution on each dataset separately using the maximum likelihood method explained below. We compared the extracted effective rates to the expected diffusion-limited particle binding rate (Fig. S2F), $k = \kappa c$, with

$$\kappa = \frac{2 k_\mathrm{B} T R_\mathrm{cell}}{3\eta R_\mathrm{particle}} \approx 0.2 \  \mathrm{pM^{-1}\ s^{-1}},$$

where $\eta = 0.7\ \mathrm{mPa\ s}$ is the viscosity of water at $T = 37^{\circ}\mathrm{C} = 310\ \mathrm{K}$ [1], $R_\mathrm{cell} = 5\ \mathrm{\mu m}$ is the typical B cell radius, and $R_\mathrm{particle} = 60\ \mathrm{nm}$ is the typical radius of virus-like particles [2].

### Hit models

After observing that $k_\mathrm{obs}$ was close to this diffusion limit at low particle concentrations but well below it at higher concentrations, we developed refined models incorporating particle hit steps and an intrinsic cell activation step. 

Specifically, we considered an $M$-hit model in which a B cell must first encounter $M$ particles sequentially, each with hit rate $k_1$ (concentration-dependent), before transitioning to the active state with a concentration-independent intrinsic rate $k_2$. The B cell therefore transitions through $M+1$ states in total: the first $M$ transitions occur with rate $k_1$ and the final one, with rate $k_2$. A schematic of this model appears in Fig. S2C. Let $P(m, t)$ denote the probability that the cell has encountered $m$ particles by time $t$, and $P(A, t)$ the probability that it has reached the activated state. The chemical master equation is

$$\frac{dP(m, t)}{dt} = k_1 P(m-1, t) - k_1 P(m, t), \quad 0 \leq m < M,$$
$$\frac{dP(M, t)}{dt} = k_1 P(M-1, t) - k_2 P(M, t),$$
$$\frac{dP(A, t)}{dt} = k_2 P(M, t),$$

with the convention $P(-1, t) = 0$ in the first equation for $m=0$. The initial condition is $P(0, 0) = 1$. The CDF $F_{t^{\star}}$ is the probability that the cell has reached the final state $A$ by time $t$; therefore, $F_{t^{\star}}(t) = P(A, t)$. The associated probability density function (PDF) is, from the master equation, $f_{t^{\star}}(t) = \frac{dF_{t^{\star}}(t)}{dt} = k_2 P(M, t)$. 

Solving this system of linear differential equations in Laplace space, we find, in general, that the PDF and CDF of a $M$-hit model are given by the inverse Laplace transform of

$$\tilde{f}(s) = \frac{k_1^M k_2}{(k_1 + s)^M (k_2 + s)} \quad \text{and} \quad \tilde{F}(s) = \frac{k_1^M k_2}{s (k_1 + s)^M (k_2 + s)} \ \  .$$

For the 1-, 2-, and 3-hit models, assuming that $k_1 \neq k_2$, these inverse transforms yield 

**One-hit model:**
$$f_{t^{\star}}(t) = \frac{k_1 k_2}{k_2 - k_1} \left(e^{-k_1 t} - e^{-k_2 t} \right) \qquad \text{and} \qquad F_{t^{\star}}(t) = 1 + \frac{k_1 e^{-k_2 t} - k_2 e^{-k_1 t}}{k_2 - k_1}$$

**Two-hit model:**
$$f_{t^{\star}}(t) = \frac{k_1 k_2}{k_2 - k_1} \left[ k_1 t e^{-k_1 t} - \frac{k_1}{k_2 - k_1} \left(e^{-k_1 t} - e^{-k_2 t} \right) \right]$$
$$F_{t^{\star}}(t) = \frac{k_2}{k_2 - k_1} \left[ 1 - (1 + k_1 t)e^{-k_1 t} - \frac{k_1^2}{k_2 - k_1} \left( \frac{1 - e^{-k_1 t}}{k_1} - \frac{1 - e^{-k_2 t}}{k_2} \right) \right]$$

**Three-hit model:**
$$f_{t^{\star}}(t) = \frac{k_1^3 k_2}{k_2 - k_1} \left[ \frac{t^2 e^{-k_1 t}}{2} - \frac{t e^{-k_1 t}}{k_2 - k_1} + \frac{e^{-k_1 t} - e^{-k_2 t}}{(k_2 - k_1)^2} \right]$$
$$F_{t^{\star}}(t) = 1 - \frac{k_2 k_1^2 t^2 e^{-k_1 t}}{2(k_2 - k_1)} + \frac{k_1^3 e^{-k_2 t}}{(k_2 - k_1)^3} + \frac{(2k_1 - k_2) k_2 k_1 t e^{-k_1 t}}{(k_2 - k_1)^2} - \frac{k_2(3k_1^2 - 3k_2 k_1 + k_2^2) e^{-k_1 t}}{(k_2 - k_1)^3}$$

To fit these models to experimental data where the true time origin is uncertain, we introduced an additional parameter $t_0$ (required to be smaller than the first observed activation time). This was implemented by replacing $t$ with $t - t_0$ throughout the PDF and CDF expressions.

## Maximum likelihood parameter estimation

We fit these models to the empiric CDF of activated cells over time, treating individual B cells as independent realizations of the same first-passage process, and inferred parameters via maximum likelihood. We used a likelihood function including a PDF term for each activation event up to a cutoff time $t_\text{cut}$ — chosen as the time at which the empiric CDF reaches 75% active cells — and a CDF term for later events. This cutoff reduced sensitivity to spurious inactive tracks not corrected by our eCDF normalization while retaining the majority of activation events. 

We inferred one global cell-intrinsic activation rate $k_2$ across all datasets, and one pair $(k_1, t_0)$ per dataset to account for varying concentrations and replicate variability. We therefore maximized a total log-likelihood $\log \mathcal{L}_\mathrm{tot} = \sum_{j} \log \mathcal{L}_j$ across datasets, where the log-likelihood contribution of a dataset is defined as follows. 

For recording $j$ containing $N$ cells with valid calcium traces, suppose $K_0$ tracks are already active at $t = 0$, $K_1$ become activated at times $\{t_{K_0+1}, \ldots, t_{K_0+K_1}\}$ between $t = 0$ and $t_\text{cut}$, and the remaining $N - K_0 - K_1$ cells are still inactive at $t_\text{cut}$. Initially active cells contribute $F_{t^{\star}}(0)$ to the likelihood, and cells still inactive at $t_\text{cut}$ contribute $1 - F_{t^{\star}}(t_\text{cut})$. The log-likelihood for dataset $j$ is therefore 

$$\log \mathcal{L}_j = K_0 \log\!\left(F_{t^{\star}}(0)\right) + \sum_{n=K_0+1}^{K_0+K_1} \log\!\left(f_{t^{\star}}(t_n)\right) + \left(N - K_0 - K_1\right) \log\!\left(1 - F_{t^{\star}}(t_\text{cut})\right).$$

For each hit model, we maximized $\log \mathcal{L}_\mathrm{tot}$ across datasets with respect to the global rate $k_2$ and the per-recording parameters $(k_1, t_0)$, using the interior-point Newton algorithm in *Optim.jl* (Julia) [3, 4]. Parameters $k_2$ and $k_1/c$ (where $c$ is the particle concentration in pM) were constrained to $[10^{-4},\  20.0]\ \text{s}^{-1}$, and $t_0$ was bounded between $-20\ \text{s}$ and $t_1 - 10^{-3}\ \text{s}$, where $t_1$ is the first observed activation time. For Fig. S2G-I, we repeated this process across a range of fluorescence threshold factors $f$. The fits for $k_1$ were generally insensitive to $f$, whereas $t_0$ varied more because the earliest activation time $t_1$ is delayed as $f$ increases. 


## Comparison of the different hit models

At high particle concentrations, the hit rate $k_1$ is fast and the intrinsic activation step with rate $k_2$ limits activation; at low concentrations, the converse holds. Consequently, the global value of $k_2$ is mainly constrained by high-concentration datasets, where the 1-, 2-, and 3-hit models are essentially indistinguishable. We therefore compared model accuracy only for the four datasets at 0.01 pM and 0.1 pM (two replicates each), where the models are most clearly distinguished and $k_1$ is close to the diffusion limit. Fit quality was quantified by the root-mean-squared (RMS) error on the CDF up to $t_\mathrm{cut}$. 


## References

[1] Grégoire Altan-Bonnet, Thierry Mora, Aleksandra M. Walczak. "Quantitative Immunology for Physicists", *Physics Reports*, 2020, vol. 849, pp. 1--83. https://doi.org/10.1016/j.physrep.2020.01.001

[2] Jeremy F. Brooks, Julianne Riggs, James L. Mueller, Raisa Mathenge, Wei-Yun Wholey, Alexander R. Meyer, Sekou-Tidiane Yoda, Vivasvan S. Vykunta, Hailyn V. Nielsen, Wei Cheng, Julie Zikherman. "Molecular basis for potent B cell responses to antigen displayed on particles of viral size", *Nature Immunology*, 2023, vol. 24, pp. 1762--1777. https://doi.org/10.1038/s41590-023-01597-9

[3] Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral B. Shah. "Julia: A fresh approach to numerical computing", *SIAM Review*, 2017, vol. 59, no. 1, pp. 65--98. https://epubs.siam.org/doi/10.1137/141000671

[4] Patrick Kofod Mogensen and Asbjørn Nilsen Riseth. "Optim: A mathematical optimization package for Julia", *Journal of Open Source Software*, 2018, vol. 3, no. 24, p. 615. https://doi.org/10.21105/joss.00615