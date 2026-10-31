# Assignment 11 — Optimizers and Learning-Rate Schedules, Measured

A gradient tells each weight which way to move, but not how far. Session 11 is about the rules for the distance:
Adam, bias correction, warmup, schedules and learning-rate transfer across widths. This assignment checks each rule on
a real model and real data. Every claim is predicted by hand first and then measured, and every comparison is tuned
on both sides.

- **Model:** the Llama-style decoder from Assignment 10, written from scratch ([`model.py`](model.py): RMSNorm, RoPE,
  SwiGLU). The head is now a separate matrix so that muP can treat it differently. Default 49.3M weights
  (L=6, D=384); width sweep at D = 256, 512, 1024 with 4 layers (29M to 154M weights).
- **Data:** TinyStories, 40,000 stories, GPT-2 BPE, packed into one 9.0M-token stream (5% held out). Batches are
  32 × 256 = 8,192 tokens; held-out loss is measured on the same 65,536 tokens every time.
- **Loop:** AdamW (and a hand-written `HandAdamW` with a bias-correction switch), bf16 autocast, gradient clipping
  at 1.0, 300-step runs. The same seed means the same initial weights and the same batches, so compared runs differ
  only in the thing being compared.
- **Hardware:** Google Colab Pro, NVIDIA A100-SXM4-40GB, PyTorch 2.11. The whole notebook (about 85 training runs)
  takes 21.5 minutes.

Notebook: [`session11_optimizers.ipynb`](session11_optimizers.ipynb). It runs top to bottom with every output saved.
Every number below is copied from that run.

## Results

| Task | Result |
|---|---|
| 1. Adam by hand vs PyTorch, lesson's 5 steps | identical $w$ in float64 (difference 0), $\le 3.1\times10^{-8}$ in float32 |
| 1. Same check on a real 442,368-weight matrix, real gradients | max difference $1.49\times10^{-8}$ (one float32 spacing) vs steps of $10^{-3}$ |
| 2. Bias correction off: uncorrected / corrected step ($\beta_2 = 0.999$) | 3.16 at step 1, **6.57 at step 12**, 6.24 at step 20 |
| 2. Steps until the difference stops mattering (within 1%) | **3,925** for $\beta_2 = 0.999$ · **76** for $\beta_2 = 0.95$ (real model: within noise from step 63) |
| 2. Real model, 300 steps, $\beta_2 = 0.999$, held-out loss | corrected 3.231 · **uncorrected 4.211 (+0.98)** · with warmup 100 still +0.67 |
| 3. Update/weight ratio at step 1 (no warmup), hidden matrices | predicted $\eta/\sigma$ = 0.01960, measured 0.01957: the lesson's "19× too large" |
| 3. Step at which warmup stops changing the ratio | about **step 113 to 131** for warmup 50 and 100 alike (embedding: right after warmup; `mlp.down`, head: never in 300 steps) |
| 4. Cosine vs WSD, planned 300, stopped at 200 (3 seeds) | cosine 3.863 ± 0.044 · **WSD 3.702 ± 0.035**, paired gap −0.161 ± 0.014 |
| 4. Model kept | **the WSD step-200 checkpoint** (lower loss, and it can continue or be decayed later) |
| 5. SP optimal learning rate at widths 256 / 512 / 1024 | $1.91\times10^{-3}$ / $7.60\times10^{-4}$ / $4.16\times10^{-4}$, fitted exponent **−1.10** (theory −1) |
| 5. Value for width 4,096 | **$1\times10^{-4}$**, low to moderate confidence (factor 2 either way, for 300-step runs) |
| 5. muP sweep: spread of the optimum across widths | **1.9×** (SP: 4.6×) |
| Extra: Adam step for noisy gradients | 0.183 η at steady state (lesson: 0.281 η, not reproduced) |
| Extra: L2 vs AdamW decay, small-gradient weight vs large | L2 **17×** stronger (lesson: 100×), AdamW 1.00× |
| Extra: AdamW memory, measured | 4 + 4 + 8 = **16 bytes per weight**; 9B model 134.1 GiB, the lesson's figure |

## Task 1 — Adam by hand

Adam keeps two running averages per weight, $m$ for the gradient and $v$ for its square, corrects their start-at-zero
bias and steps by their ratio:

$$m \leftarrow \beta_1 m + (1-\beta_1) g,\quad v \leftarrow \beta_2 v + (1-\beta_2) g^2,\quad
\hat m = \frac{m}{1-\beta_1^t},\ \hat v = \frac{v}{1-\beta_2^t},\quad w \leftarrow w - \eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

By hand, step 1 of the lesson's example ($w_0 = 1$, $g = 0.5$, $\eta = 10^{-3}$): $m = 0.05$, $v = 0.00025$,
$\hat m = 0.5$, $\hat v = 0.25$, step $= -0.001$, so the first step is exactly $-\eta\,\text{sign}(g)$. The
plain-Python version reproduces the lesson's five-step table to six decimals (0.999000 … 0.995031). Every step is
within 1.19% of $\eta$ even though the gradients vary by 50%.

| check | max \|hand − PyTorch\| in $w$ |
|---|---:|
| `torch.optim.Adam`, float64 | 0 |
| `torch.optim.AdamW` (λ = 0.1), float64 | 0 |
| `Adam` / `AdamW`, float32 | 1.3e-8 / 3.1e-8 (below one float32 spacing near 1, 1.19e-7) |
| `HandAdamW` vs `AdamW` on the real `blocks.0.attn.qkv` (442,368 weights, 5 real gradients) | 1.49e-8 = $2^{-26}$ |

On real gradients the first step was $-0.99975\,\eta$ rather than exactly $-\eta$. That weight's gradient is about
$4\times10^{-5}$, so $\epsilon = 10^{-8}$ is no longer negligible. The mean step over the matrix falls from
$9.99\times10^{-4}$ to $3.48\times10^{-4}$ within 5 steps: the gradients stop agreeing with each other almost at once.

**Why "Adam moves every weight by about η" is only half true** ([plot](adam_step_vs_consistency.png)). The step
depends on how consistent the gradient is, not on its size:

| gradient sequence | predicted $\lvert\text{step}\rvert/\eta$ | measured (steady state) |
|---|---:|---:|
| constant | 1 | 1.000 |
| alternating ±1 | $(1-\beta_1)/(1+\beta_1) = 0.053$ | 0.053 |
| noise, mean 0 | $\sqrt{0.1/1.9}\sqrt{2/\pi} = 0.183$ | 0.183 |

The lesson quotes 0.281 η for the noisy case. Averaged over the first 10 / 100 steps we get 0.448 / 0.216, so its
number depends on an averaging window it does not state.

**L2 vs decoupled decay.** With noisy gradients (std 1 for weight A, 0.01 for weight B), AdamW shrinks both by
$4.93\times10^{-5}$ per step ($=\eta\lambda w$). L2 shrinks B by $8.37\times10^{-4}$ and A by $4.90\times10^{-5}$:
**17 times more, not the lesson's 100.** B's decay term ($\lambda w = 0.05$) is larger than its gradient noise, so it
takes over both $\hat m$ and $\hat v$, and Adam can never move a weight much more than $\eta$.

**Memory.** Measured on the GPU: weight 4 + gradient 4 + AdamW state 8 = 16 bytes per weight (SGD + momentum 12,
SGD 8). For 9B weights that is 134.1 / 100.6 / 67.1 GiB, the lesson's table exactly. Half the AdamW bill is $m$ and $v$.

## Task 2 — Bias correction switched off

Without correction the update uses $m/\sqrt v$. Since $m = \hat m(1-\beta_1^t)$ and $v = \hat v(1-\beta_2^t)$:

$$\frac{m}{\sqrt v} = \frac{\hat m}{\sqrt{\hat v}}\cdot c(t),\qquad c(t) = \frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}}$$

So the uncorrected step is the corrected one times a factor that depends **only on the step count**. "When does it
stop mattering" becomes a question with an exact answer: when $c(t)$ is close to 1.

![bias factor](bias_correction_factor.png)

| $\beta_2$ | $c(1)$ | peak | $c(20)$ | within 10% from | within 1% from |
|---:|---:|---:|---:|---:|---:|
| 0.999 | 3.162 | **6.569 at step 12** | 6.241 | 1,751 | **3,925** |
| 0.99 | 1.000 | 2.131 at step 13 | 2.059 | 175 | 391 |
| 0.95 | 0.447 | 1.097 at step 20 | 1.097 | 6 | **76** |

With $\beta_2 = 0.999$ the uncorrected step does not just start too large: it *grows* to 6.6 times too large. $m$'s
bias is gone after 44 steps, but $v$'s takes thousands. With $\beta_2 = 0.95$ the uncorrected first step is *smaller*,
so missing correction acts like a tiny warmup.

**The first 20 steps of two real weights** (their recorded gradients fed to both versions). The step ratio follows
$c(t)$: 3.15 / 6.55 / 6.22 at steps 1 / 12 / 20 for one weight and 3.04 / 6.44 / 6.13 for the other, whose gradients
are small enough ($10^{-7}$ to $10^{-5}$) for $\epsilon$ to matter. After 20 steps the uncorrected weight had moved
**5.5 times further** (0.0588 vs 0.0106). With $\beta_2 = 0.95$ the two are almost the same (0.0102 vs 0.0114).
Within 20 steps, with $\beta_2 = 0.999$, the difference never stops mattering.

![first 20 steps](bias_correction_20_steps.png)

**On the real model (300 steps, η = 1e-3).** I expected a loss spike; there was none, and the result went the other
way at first. For about 25 steps the uncorrected run is *ahead*, by up to 1.5 nats near step 5, because larger steps
help while the loss falls from 10.8. After step ~30 it falls behind for good:

| run | held-out loss at 300 | gap |
|---|---:|---:|
| $\beta_2 = 0.999$, corrected / uncorrected | 3.231 / 4.211 | **+0.980** |
| same, with 100-step warmup | 3.051 / 3.722 | +0.671 |
| $\beta_2 = 0.95$, corrected / uncorrected | 3.243 / 3.266 | +0.024 (within noise from step 63) |
| seed-to-seed, corrected | 3.231 / 3.249 | 0.018 |

$c(t)$ is still 3.24 at step 100 and 1.96 at step 300, so the uncorrected run trains at 2 to 6.6 times the intended
rate the whole time. Warmup cannot fix that: it only scales the first 100 steps, while $c(t)$ stays above 2 until
about step 290. In practice, bias correction matters because of $\beta_2 = 0.999$. With the $\beta_2 = 0.95$ common in
LLM training, missing correction is harmless after about 76 steps.

![real model](bias_correction_real_model.png)

## Task 3 — Update-to-weight ratio per layer, and warmup

$$r_t = \frac{\lVert W_t - W_{t-1}\rVert}{\lVert W_{t-1}\rVert} \approx \frac{\eta_t\,k_t}{\sigma}$$

Adam moves each element by about $\eta k$, where $k \le 1$ measures gradient consistency. A tensor of $n$ elements
with spread $\sigma$ has $\lVert W\rVert = \sigma\sqrt n$, and the $\sqrt n$ cancels. So $r_t$ is the learning rate in
units of the tensor's own scale. **During warmup $\eta_t$ forces it upward. After warmup only $k_t$ (and $\sigma$) move
it.** The task's step is the hand-off between those two regimes.

**Step 1, no warmup:** $k = 1$ exactly, so $r_1 = \eta/\sigma$. Predicted vs measured:

| tensor | σ | predicted | measured |
|---|---:|---:|---:|
| attn.qkv / mlp.up / mlp.gate | 0.051 | 0.01960 | 0.01957 |
| attn.proj | 0.0147 | 0.06783 | 0.06781 |
| mlp.down | 0.0090 | 0.1108 | 0.1107 |
| norm gains | 1 | 0.00100 | 0.00100 |
| embedding (1,244 of 50,304 rows in the batch) | 0.02 | 0.00786 | 0.00785 |
| head | 0.0026 | 0.384 | 0.381 |

That is the lesson's "nineteen times too large" first step (0.0196 vs the $10^{-3}$ rule). With warmup $W$, step 1 is
$1/W$ of it: $3.9\times10^{-4}$ ($W = 50$), $1.9\times10^{-4}$ ($W = 100$).

![ratio per layer](update_ratio_per_layer.png)

Three runs, warmup 0 / 50 / 100, then constant $\eta$ (no decay, so nothing else moves the ratio):

- **Warmup stops changing the ratio at about step 113 to 131** (first step after which the warmup run stays within
  10% of the no-warmup run) for qkv, gate, up and norm gains, **for both warmup lengths**. For warmup 50 the ratio
  also peaks right at the end of warmup (steps 52 to 58). The embedding rejoins immediately after warmup (53, 109).
- **`mlp.down` and the head never rejoin within 300 steps.** They end 10 to 25% higher in the warmup runs. They have
  the smallest initial scale, and without warmup their first steps were 11% and 38% of their own size, which
  permanently enlarged $\lVert W\rVert$.
- **$r_t/\eta_t$ (the part warmup does not set) falls from about 20 to about 4 by step 150 in every run.** Early
  gradients agree and later ones do not. This is the reason warmup exists, measured directly. Without warmup the
  ratio falls from $2\times10^{-2}$ to about $4\times10^{-3}$ in roughly 20 steps by itself.
- Settled ratios at $\eta = 10^{-3}$: $3.5$ to $4\times10^{-3}$ (qkv, gate, up), $10^{-2}$ (proj), $1.3\times10^{-2}$
  (down), $2\times10^{-4}$ (norm gains). That is 4 to 13 times the $10^{-3}$ rule, which is meant for long runs.
- Warmup also helped the loss: 3.257 → 3.206 → **3.037** for warmup 0 → 50 → 100. Section 2 shows the same.

![ratio vs warmup](update_ratio_vs_warmup.png)

## Task 4 — Cosine vs WSD, stopped at step 200

Both planned for 300 steps (20 warmup). At step 200 cosine is at 0.355 of its peak, while WSD is still flat (decay
over 241 to 300). The peak rate was tuned separately for each over {5e-4, 1e-3, 2e-3, 4e-3}; both chose 1e-3. Then 3
seeds per configuration, plus three extra configurations at equal compute:

| model | rate area to step 200 | held-out loss at 200 | at 300 |
|---|---:|---:|---:|
| cosine planned 300, stopped at 200 | 145 | 3.863 ± 0.044 | 3.580 ± 0.039 |
| **WSD planned 300, stopped at 200** | 190 | **3.702 ± 0.035** | 3.248 ± 0.031 |
| cosine planned for 200 | 109 | 4.164 ± 0.041 | — |
| WSD branch: flat to 160, decay 160 to 200 | 172 | 3.762 ± 0.043 | — |
| constant rate | 190 | 3.702 ± 0.035 | 3.238 ± 0.023 |
| constant rate, EMA of weights (decay 0.9) | | 3.705 ± 0.038 | **3.205 ± 0.025** |

Paired by seed, WSD − cosine = **−0.161 ± 0.014** at step 200 (all three seeds negative) and −0.332 ± 0.010 at 300.

**I would keep the WSD model.** It has the lower loss, and it is a flat-phase checkpoint, so it can be trained
further or decayed later. The cosine model is half-decayed from a run that was cut short.

**This contradicts my prediction**, which was that cosine's partial annealing would make it better at step 200. In
this regime every decay cost loss, and the loss at step 200 is ordered exactly by how much learning rate each run has
used. Even the "finished" WSD branch is worse than not decaying, and WSD's own decay over steps 240 to 300 gained
nothing over a constant rate. After only 2.5M tokens the model is far from converged: the loss is still falling by
0.25 nats per 50 steps. Progress is limited by how far the weights can travel, not by gradient noise, so decaying the
rate hurts. The annealing drop the lesson describes needs a longer run. A short weight average (EMA 0.9) gave the best
step-300 model. Longer averages (0.98) hurt, because the weights move fast.

![cosine vs WSD](cosine_vs_wsd.png)

## Task 5 — Learning-rate sweep across widths

With Adam each element of a hidden matrix moves by about $\eta$ whatever the width. A layer's output sums $d$ such
changes coherently, $\Delta y \approx d\,\eta\,|x|$, so in the standard parameterization (SP) $\eta^* \propto 1/d$.
**muP** builds that $1/d$ into the model (hidden rate $\eta/m$, logits divided by $m$, $m = d/256$), so one $\eta$
should be optimal at every width.

9 rates from 1e-4 to 5e-2, 300 steps, cosine with 30 warmup steps; minima located by a parabola in $\log\eta$:

| width | SP $\eta^*$ | SP loss | muP $\eta^*$ | muP loss | seed std at the best point |
|---:|---:|---:|---:|---:|---:|
| 256 | 1.91e-3 | 3.457 | 1.91e-3 (same model) | 3.457 | 0.092 |
| 512 | 7.60e-4 | 3.507 | 3.02e-3 | 3.553 | 0.082 |
| 1024 | 4.16e-4 | 3.599 | 3.66e-3 | 3.642 | 0.081 |

![LR sweep](lr_sweep_sp_vs_mup.png)

**SP follows the theory.** Fitted law $\eta^* = 0.80\,d^{-1.10}$, against the theoretical exponent −1.

**Value for width 4,096: $1\times10^{-4}$.** The fitted law gives $8.6\times10^{-5}$ and the −1 rule anchored at
width 1,024 gives $1.04\times10^{-4}$. The lesson's table says $1.9\times10^{-4}$ for its own setting.

**Confidence: low to moderate. I would trust it to a factor of 2 (5e-5 to 2e-4) for a 300-step run.**
- Seed noise at the optimum (0.08 to 0.09 nats) is as large as the gap between neighbouring grid points, so each
  minimum is uncertain by about one grid step.
- With 3 points and 2 fitted numbers the formal 95% interval is $1.6\times10^{-6}$ to $4.7\times10^{-3}$. The estimate
  is believable mainly because the slope matches theory.
- It is a 4× extrapolation, and short runs favour larger rates.

**muP:** the optimum's spread across widths drops from 4.6× (SP) to 1.9×. It is better but not perfect: the minima
drift upwards by about 1.4× per doubling, within the noise of each minimum, and possibly also due to my minimal muP.
It does not make the model better (losses within noise of SP), which is the lesson's point. **Cross-check:** if muP
transfer holds, the hidden matrices at width 4,096 would get about $(2\text{ to }4)\times10^{-3}/16 =
1.2\text{ to }2.3\times10^{-4}$, within a factor of 2 of the SP extrapolation.

![extrapolation](lr_extrapolation.png)

## Limits of this work

- **Short runs, small models.** 300 steps (2.5M tokens) on 29M to 154M weights, 67 to 89% of them embedding and head.
  The loss is still falling fast at the end. The schedule result in Task 4 describes this undertrained regime and is
  not a general ranking of cosine and WSD. Long-run effects (AdamW's $1/(\eta\lambda)$ averaging timescale) cannot
  appear here.
- **Few seeds, large noise.** Seed spread is 0.02 to 0.04 nats in Tasks 2 to 4 and 0.08 to 0.09 nats at the sweep
  optima. Task 4 tunes the rate on one seed. Task 5 runs one seed per grid point plus 2 extra seeds at each optimum.
- **Width 4,096 is extrapolated** from 3 points, 4 times beyond the widest model.
- **Thresholds are my choices:** 1% / 10% for "stops mattering", a 10% agreement band for the warmup hand-off, loss
  above 15 as "diverged" (at width 256, 2e-2 was stopped at a single spike while 5e-2 survived), three-point
  parabolas for the minima.
- **muP is minimal:** the embedding rate and weight decay are not rescaled with width; head width is fixed at 64.
- **Not reproduced:** the lesson's 0.281 η noisy-gradient step and its 100× L2 gap. Both differences are explained
  above, but I cannot tell what setting the lesson used.

## A note on compute

While doing this course I once left a Colab runtime connected after a run had finished and lost about 30% of a
month's credits to an idle GPU. That is a small version of a real failure in LLM training: expensive accelerators
held by a job that has stopped making progress. So this notebook ends with a cell that releases the runtime
(`google.colab.runtime.unassign()`) as soon as the last result is printed.

## How to run

1. Open [`session11_optimizers.ipynb`](session11_optimizers.ipynb) in Colab (or VS Code connected to a Colab
   kernel) with a GPU. An A100 runs it in about 22 minutes.
2. Run all cells. The notebook installs `tiktoken`, writes `model.py`, downloads one TinyStories shard (no HF
   token needed) and trains every model it reports.
3. The last cell releases the runtime. In VS Code, save the notebook afterwards to keep the outputs.

Files: `session11_optimizers.ipynb` (all tasks, executed), `model.py` (the model), and the PNG plots extracted from
the notebook.
