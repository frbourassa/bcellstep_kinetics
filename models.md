# Population kinetics analysis of B cell activation by VLPs

## Calcium traces processing for Fig. S2

For each recording (defined by VLP concentration and replicate), we first filtered out spurious calcium fluorescence traces, excluding traces lasting fewer than 50 frames or beginning more than 30 s after the start of the recording. We retained only replicates containing at least 15 cells, to ensure meaningful fits could be performed.

To distinguish inactive from active B cells, we defined a calcium fluorescence intensity threshold. We set the threshold as a factor $f$ times the background fluorescence; the background was itself determined by finding the minimum fluorescence in each trace, then taking the median over traces. We used the same factor $f$ across recordings. We tested a range of $f$ values between 1.1 and 2.5 and settled on a default value $f = 1.8$ (Fig. S2G-I).

To determine the activation time $t^{*}$ of each B cell (i.e. the time at which its calcium trace first crossed the threshold), we performed linear interpolation between the last time point below threshold and the first time point above it. Traces already above threshold at the start of imaging were assigned $t^{*} = 0$. Cells that never crossed the threshold were labeled inactive for the entire recording. Representative calcium fluorescence traces and the thresholding procedure are illustrated in Fig. S2D.


## Empiric cumulative distribution function (CDF) normalization

In principle, tracks that do not reach activation during a recording carry information about the activation time distribution; however, in practice, several of those tracks come from anergic, dead, or out-of-focus B cells. To estimate the fraction of such inactive tracks to reject, we reasoned that, in high (10 pM) particle concentrations, all B cells should have had enough time to encounter several particles and activate. We thus used the fraction of inactive tracks in those datasets as an estimate of the fraction of tracks to reject, and removed at most this fraction of inactive tracks from other datasets. This still left, for instance, $\sim 25 \, \%$ inactive tracks at the end of the recordings for 0.01 pM concentration, where the particle contact rate is low enough that several cells have not yet been activated. 


## Mathematical modeling of the distribution of B cell activation times

We treated B cell activation as a stochastic process governed by the waiting time for a B cell to encounter one or more randomly diffusing VLPs and subsequently reach the calcium activation threshold. The activation time $t^{*}$ thus corresponds to the first-passage time (FPT) for the B cell to reach the "active" state. The cumulative distribution function of the FPT, $F_{t^{*}}(t) = P(t^{*} < t)$, gives the probability that activation has occurred at or before time $t$.

### Effective activation rate
First, we extracted an ``observed'' activation rate, for each experimental replicate using a one-step activation model, where a B cell transitions from inactive to active with an effective rate $k_\mathrm{obs}$ coarse-graining all steps to activation. In this simplest possible model of activation, the first-passage time has an exponential distribution and thus a distribution

$$f_{t^{*}}(t) = k e^{-kt} \qquad \text{and} \qquad F_{t^{*}}(t) = 1 - e^{-kt} \,\, .$$

We compared those effective rates to the expected diffusion-limited particle binding rate (Fig. S2F), $k = \kappa c$ with $\kappa = \frac{2 k_\mathrm{B} T R_\mathrm{cell}}{3\eta R_\mathrm{particle}} \approx 0.16 \, \mathrm{pM^{-1} s^{-1}}$, where $\eta = 0.7 \mathrm{mPa \, s}$ is the viscosity of water at $T = 37 \, {}^{\circ}\mathrm{C} = 310 \, \mathrm{K}$ [1], $R_\mathrm{cell} = 4 \mathrm{\mu m}$ is the typical radius of a B cell, and $R_\mathrm{particle} = 60 \, \mathrm{nm}$ is the typical radius of the virus-like particles used in the experiments [2]. 

### Hit models
After observing that $k_\mathrm{obs}$ was near this limit at low particle concentrations, but significantly below at higher concentrations, we developed refined models including particle hit steps and an intrinsic cell activation step. 

Specifically, we considered a $M$-hit model where a B cell must first encounter $M$ particles sequentially, each with the same hit rate $k_1$ that can depend on concentration, before transitioning to a final active state with rate $k_2$ independent from concentration (only accessible after $M$ hits). In other words, the $M$-hit model entails that the B cell must sequentially transition through $M+1$ states in total, the first $M$ with rate $k_1$ and the last one with rate $k_2$. A schematic of this model appears in Fig. S2C. Let $P(m, t)$ denote the probability that the cell has encountered $m$ particles at time $t$, and $P(A, t)$ denote the probability that it has reached the activated state. The chemical master equation is

$$\frac{dP(m, t)}{dt} = k_1 P(m-1, t) - k_1 P(m, t) \quad 0 \leq m < M $$
$$\frac{dP(M, t)}{dt} = k_1 P(M-1, t) - k_2 P(M, t)  $$
$$\frac{dP(A, t)}{dt} = k_2 P(M, t)  $$

with the convention $P(-1, t) = 0$ in the first equation for $m=0$. The initial condition is $P(0, 0) = 1$. The CDF $F_{t^{*}}$ is the probability that the cell has reached the final state $A$ by time $t$; therefore, $F_{t^{*}}(t) = P(A, t)$. The associated probability density function (PDF) is, from the master equation,

$$f_{t^{*}}(t) = \frac{dF_{t^{*}}(t)}{dt} = k_2 P(M, t).$$

Solving this system of linear differential equations in Laplace space, we find, in general, that the PDF and CDF of a $M$-hit model are given by the inverse Laplace transform of
$$\tilde{f}(s) = \frac{k_1^M k_2}{(k_1 + s)^M (k_2 + s)} \quad \text{and} \quad \tilde{F}(s) = \frac{k_1^M k_2}{s (k_1 + s)^M (k_2 + s)} \,\, .$$

For the 1-, 2-, and 3-hit models under consideration, this corresponds to

**One-hit model:**
$$f_{t^{*}}(t) = \frac{k_1 k_2}{k_2 - k_1} \left(e^{-k_1 t} - e^{-k_2 t} \right)  \qquad \text{and} \qquad F_{t^{*}}(t) = 1 + \frac{k_1 e^{-k_2 t} - k_2 e^{-k_1 t}}{k_2 - k_1}$$

**Two-hit model:**
$$f_{t^{*}}(t) = \frac{k_1 k_2}{k_2 - k_1} \left[ k_1 t e^{-k_1 t} - \frac{k_1}{k_2 - k_1} \left(e^{-k_1 t} - e^{-k_2 t} \right) \right] \quad \text{and} $$

$$ F_{t^{*}}(t) = \frac{k_2}{k_2 - k_1} \left[ 1 - (1 + k_1 t)e^{-k_1 t} - \frac{k_1^2}{k_2 - k_1} \left( \frac{1 - e^{-k_1 t}}{k_1} - \frac{1 - e^{-k_2 t}}{k_2} \right) \right] $$

**Three-hit model:**
$$f_{t^{*}}(t) = \frac{k_1^3 k_2}{k_2 - k_1} \left[ \frac{t^2 e^{-k_1 t}}{2} - \frac{t e^{-k_1 t}}{k_2 - k_1} + \frac{e^{-k_1 t}}{(k_2 - k_1)^2} - \frac{e^{-k_2 t}}{(k_2 - k_1)^2} \right] \quad \text{and} $$
$$F_{t^{*}}(t) = 1 - \frac{k_2 k_1^2 t^2 e^{-k_1 t}}{2 (k_2 - k_1)} + \frac{k_1^3 e^{-k_2 t}}{(k_2 - k_1)^3} + \frac{(2 k_1 - k_2) k_2 k_1 t e^{-k_1 t}}{(k_2 - k_1)^2} - \frac{k_2 (3 k_1^2 - 3 k_2 k_1 + k_2^2) e^{-k_1 t}}{(k_2 - k_1)^3} $$


To fit these models to experimental data where the true time origin is uncertain, we introduced an additional parameter $t_0$ (required to be smaller than the first observed activation time). This was implemented by replacing $t$ with $t - t_0$ throughout the PDF and CDF expressions.

## Maximum likelihood parameter estimation

We fit these models to the empiric CDF of activated cells over time, treating individual B cells as independent realizations of the same first-passage process. We inferred parameters using a maximum likelihood approach. We used a likelihood function including a PDF term for events up to a cutoff time $t_\text{cut}$, chosen where the empiric CDF reaches 75% active cells, and including later events in a CDF term contribution. This cutoff time reduced sensitivity to spurious inactive tracks not corrected by our eCDF normalization while still emphasizing the majority of cell activation events (or even all of them in some cases). 

We inferred one global cell-intrinsic activation rate $k_2$ across all datasets, and one value of $k_1$ and $t_0$ for each dataset (to account for different concentrations and replicate variability). Therefore, we maximized a total log-likelihood $\log \mathcal{L}_\mathrm{tot} = \sum_{j} \mathcal{L}_j$ across datasets, combining the likelihoods $\mathcal{L}_j$ of each dataset, defined in the following way. 

Consider recording $j$ containing $N$ cells with valid calcium traces. Suppose $K_0$ of these tracks are already active at the start of the recording, $K_1$ of them become activated at times $\{t_{K_0+1}, \ldots, t_{K_0+K_1}\}$ between $t = 0$ and $t_\text{cut}$, and the $N - K_0 - K_1$ remaining cells remain inactive until $t_\text{cut}$ at least. Then, the likelihood of this dataset for a given model is the product of the probability density of the activation times that lie between 0 and $t_\text{cut}$, with additional CDF contributions from the other cells, for proper normalization. Initially active cells contribute a probability $P(t^{*} \leq 0) = F_{t^{*}}(0)$, while cells still inactive at $t_\text{cut}$ contribute a probability $P(t^{*} \geq t_\text{cut}) = 1 - F_{t^{*}}(t_\text{cut})$. Therefore, in total, the log-likelihood function contribution from dataset $j$ is

$$\log \mathcal{L}_j = K_0 \log\!\left(F_{t^{*}}(0)\right) + \sum_{n=K_0+1}^{K_0+K_1} \log\!\left(f_{t^{*}}(t_n)\right) + \left(N - K_0 - K_1\right) \log\!\left(1 - F_{t^{*}}(t_\text{cut})\right).$$


For each hit model, we maximized this total log-likelihood with respect to the global rate $k_2$ and the vector of $(k_1, t_0)$ parameters for each recording. We performed gradient-based optimization using the interior-point Newton algorithm implemented in the *Optim.jl* module in Julia [3, 4]. Parameters $k_2$ and $k_1/c$ (where $c$ is the particle concentration, in pM) were constrained between $10^{-4} \, \text{s}^{-1}$ and $20.0 \, \text{s}^{-1}$)$, and $t_0$ was bounded between $-20\ \text{s}$ and $t_1 - 10^{-3}\ \text{s}$, where $t_1$ is the first observed activation time. For Fig. S2G-I, we repeated this process for a range of fluorescence threshold factors $f$. The optimal fits for parameter $k_1$ were generally insensitive to $f$, whereas $t_0$ varied more with $f$, because the earliest activation time $t_1$ may be delayed as $f$ increases. 

## Comparison of the different hit models

After fitting the model parameters across datasets, we find that at high particle concentrations, the hit rate $k_1$ is fast and the cell-intrinsic step with rate $k_2$ limits cell activation, while at low concentrations, the cell activation step is fast compared to hit steps. As a result, the global value of $k_2$ is mainly determined by the high concentration datasets, where the 1-, 2-, and 3-hit models are essentially identical. Therefore, we compared the relative accuracy of those models in capturing B cell activation kinetics only for the four datasets at 0.01 pM and 0.1 pM concentration (two replicates of each concentration), where the models are most clearly distinguished from one another and where cell activation is limited by hit rates $k_1$ close to the diffusion limit. We quantified the fit quality in terms of the root-mean squared (RMS) error on the CDF curve up to $t_\mathrm{cut}$. 


## References

[1] Grégoire Altan-Bonnet, Thierry Mora, Aleksandra M. Walczak. "Quantitative Immunology for Physicists", *Physics Reports*, 2020, vol. 849, pp. 1--83. https://doi.org/10.1016/j.physrep.2020.01.001 

[2] Jeremy F. Brooks, Julianne Riggs, James L. Mueller, Raisa Mathenge, Wei-Yun Wholey, Alexander R. Meyer, Sekou-Tidiane Yoda, Vivasvan S. Vykunta, Hailyn V. Nielsen, Wei Cheng, Julie Zikherman. "Molecular basis for potent B cell responses to antigen displayed on particles of viral size", *Nature Immunology*, 2023, vol., 24, pp. 1762–-1777. https://doi.org/10.1038/s41590-023-01597-9

[3] Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral B. Shah. "Julia: A fresh approach to numerical computing", *SIAM Review*, 2017, vol. 59, no 1, pp. 65--98. https://epubs.siam.org/doi/10.1137/141000671

[4] Patrick Kofod Mogensen and Asbjørn Nilsen Riseth. "Optim: A mathematical optimization package for {Julia}", *Journal of Open Source Software*, 2018, vol., 3, no 24, p. 615. https://doi.org/10.21105/joss.00615