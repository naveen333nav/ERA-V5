# Session 14 — the MoE terms used in the assignment plan

All examples use tiny numbers you can check by hand. "Expert" = one small feed-forward network (FFN).
"Router" = a matrix that gives every token one score per expert. "Top-k" = keep the k best scores.

---

## 1. Balancing: none / aux loss α = 0.01 / bias γ = 1e-3, and MaxVio

**The problem they solve.** The router and the experts learn together. An expert that gets picked a bit
more often gets more training, gets better, and gets picked even more often. Left alone, a few experts
take almost all tokens and the rest never learn ("dead experts"). This is **routing collapse**.

### 1a. How we measure imbalance: MaxVio

The **load** of an expert is how many tokens it received in a batch.

$$\text{MaxVio} = \frac{\max_i \text{load}_i - \text{mean load}}{\text{mean load}}$$

Example, 4 experts, 8 tokens, each token picks 1 expert:

| loads | mean | max | MaxVio | meaning |
|---|--:|--:|--:|---|
| [2, 2, 2, 2] | 2 | 2 | **0** | perfectly even |
| [4, 2, 1, 1] | 2 | 4 | **1.0** | busiest expert has 2× the average |
| [5, 1, 1, 1] | 2 | 5 | **1.5** | |
| [8, 0, 0, 0] | 2 | 8 | **3.0** | full collapse (= number of experts − 1) |

"MaxVio over time" = we compute this every few training steps and plot it. A curve that climbs means
collapse is happening; a curve that stays near 0 means balancing works.

### 1b. Option "none"

The router picks top-k by its score, nothing else. This is the control run: we expect MaxVio to climb
and dead experts to appear.

### 1c. Option "aux loss, α = 0.01" (Switch Transformer, 2021)

Add a second term to the training loss that is small when the load is even and large when it is not:

$$L_\text{total} = L_\text{language} + \underbrace{\alpha \cdot N \cdot \sum_{i=1}^{N} f_i\, P_i}_{L_\text{aux}}$$

- $N$ = number of experts
- $f_i$ = fraction of tokens actually **sent** to expert $i$ (a count, so it has no gradient)
- $P_i$ = the **average router probability** given to expert $i$ (this has a gradient)
- $\alpha$ = how strongly we push (0.01 here)

Example with $N = 4$, $\alpha = 0.01$:

| state | $f$ | $P$ | $\sum f_i P_i$ | $L_\text{aux}$ |
|---|---|---|--:|--:|
| even | [¼, ¼, ¼, ¼] | [¼, ¼, ¼, ¼] | 4 × 1/16 = 0.25 | 0.01 × 4 × 0.25 = **0.01** |
| collapsed | [1, 0, 0, 0] | [1, 0, 0, 0] | 1 | 0.01 × 4 × 1 = **0.04** |

The gradient of $L_\text{aux}$ with respect to $P_i$ is $\alpha N f_i$, which is largest for the busiest
expert. The optimizer then lowers the router probability of busy experts and, because the
probabilities sum to 1, raises the quiet ones.

**Drawback:** this gradient goes into the same router weights the language loss is training, so the
two objectives pull against each other. Too small an α → collapse; too large → worse language loss.

### 1d. Option "bias, γ = 1e-3" (aux-loss-free balancing, Wang et al. 2024 / DeepSeek-V3)

Every expert gets one extra number, a **bias** $b_i$. It is added to the score **only for choosing**
the top-k. The weight that multiplies the expert's output still uses the original score.
The bias is **not** trained by gradient; after every step we nudge it by hand:

$$b_i \leftarrow b_i + \gamma \cdot \text{sign}(\text{mean load} - \text{load}_i)$$

- expert busier than average → $b_i$ goes **down** by γ = 0.001
- expert quieter than average → $b_i$ goes **up** by 0.001

Example, 4 experts, top-2, one token:

| expert | 0 | 1 | 2 | 3 |
|---|--:|--:|--:|--:|
| score | 0.40 | 0.35 | 0.15 | 0.10 |
| bias (after many steps, expert 0 was busy, 2 was quiet) | −0.30 | 0 | +0.25 | 0 |
| score + bias, **used to choose** | 0.10 | 0.35 | **0.40** | 0.10 |

Chosen: experts 2 and 1. Their weights come from the **original** scores: 0.15 and 0.35, rescaled to
sum to 1 → **0.30 and 0.70**.

Without the bias, experts 0 and 1 would have been chosen. The bias moved one token away from the busy
expert, and the language-loss gradient was never touched. That is the whole advantage over the aux loss.

**In the notebook** we run the same model three times (none / aux / bias), plot MaxVio and the number of
dead experts over time, and show a heatmap of the load of each expert in each layer.

**Why our router uses sigmoid scores, not softmax.** A step of γ = 0.001 only makes sense relative to the size of
the scores. Sigmoid scores lie in (0, 1) and start near 0.5, so 0.001 is a gentle 0.2% nudge (this is the setup
DeepSeek-V3 designed the method for). Softmax scores over 32 experts are about 1/32 = 0.03 each, so the same
0.001 would be about 3% of a score per step and would over-correct. For the aux losses, $P_i$ is then the sigmoid
score divided by the sum of all 32 scores, so the $P_i$ still add up to 1 as in the formula above.

---

## 2. Ways to turn the dense FFN into experts

**One fact makes all of this work.** An FFN with width $F$ is a sum over its $F$ hidden "neurons":

$$\text{FFN}(x) = \sum_{j=1}^{F} \underbrace{W^\text{down}_{:,j}}_{\text{column } j} \cdot \text{SiLU}(W^\text{gate}_{j}\,x)\cdot(W^\text{up}_{j}\,x)$$

So if you cut the neurons into groups, each group is a smaller FFN, and **the groups add back up to
the original exactly**. Tiny example: a dense FFN with 4 neurons whose contributions to the output are
$[a, b, c, d]$. Dense output = $a + b + c + d$.

The MoE output is $\sum_{i \in \text{chosen}} g_i E_i(x)$, where the router weights $g_i$ add up to 1.

Our setup: dense $F = 1024$ → MoE with **32 experts of 256 neurons, top-4**, so a token uses
4 × 256 = 1024 neurons, the same compute as the dense model.

### 2a. "copy + top-1": matches the dense model exactly at step 0

Every expert is a **full copy** of the dense FFN (all 1024 neurons). The router picks 1 expert per token,
and after rescaling a single chosen weight is $g = 1$:

$$y = 1 \cdot \text{copy of FFN}(x) = \text{FFN}(x)$$

**Hand prediction:** the validation loss right after conversion equals the dense loss (up to float
rounding), whatever the router does. The compute per token is unchanged (one 1024-wide FFN).
**Weakness:** all experts are identical, so the router cannot tell them apart. They only start to
differ because different tokens train different copies, which is slow. It also stores 1024-wide experts,
so it is not the 32 × 256 shape. We use it as the "no jump at conversion" reference.

### 2b. "partition without redraw"

Cut the 1024 neurons into 4 slices of 256 (A, B, C, D). Each slice is an expert. We need 32 experts, so
each slice is **cloned 8 times** → 8 identical A's, 8 identical B's, and so on.

With 4 neurons, think: slices $[a], [b], [c], [d]$. If the router picks one clone from each slice, the
output is $g_A a + g_B b + g_C c + g_D d$. With weights of about ¼ each, that is about **¼ of the dense
output**. We fix the scale with a **routed scaling factor** of $k = 4$ (lesson §7), so that "one of
each slice, equal weights" gives back exactly $a + b + c + d$.

But the router is new and random, so it may pick two clones of A and none of D. Then some neurons
count twice and some are missing. **Hand prediction:** a loss **jump** at conversion, smaller than with
random experts. Clones are identical, so this is where **collapse inside a clone family** can show up
(the router can't distinguish 8 identical A's, so it keeps using one or two of them).

### 2c. "drop-upcycling, r = 0.25 / 0.5 / 1.0"

Same as partition, but in every expert a fraction $r$ of its neurons are **re-drawn at random**
(their gate and up rows and their down column get fresh random weights). Each clone redraws a
**different** random subset.

Example: slice A has 4 neurons, $r = 0.5$:

| clone | kept from dense | redrawn |
|---|---|---|
| A₁ | neurons 1, 3 | 2, 4 |
| A₂ | neurons 2, 3 | 1, 4 |
| A₃ | neurons 1, 4 | 2, 3 |

So the clones start **close to the dense knowledge but different from each other**, which gives the router
something to tell apart.

| r | what is kept | expected |
|--:|---|---|
| 0 (= 2b) | everything, clones identical | smallest jump, risk of clone collapse |
| 0.25 | 75% of each expert | small jump, clones differ a little |
| **0.5** | half | the lesson's best (Drop-upcycling 2025): bigger jump, recovers faster |
| 1.0 | nothing from the FFN | experts are random; only attention and embeddings are pretrained |

The sweep runs each for the same short budget and compares (a) the loss right after conversion and
(b) the loss at the end. More redraw means you lose more at step 0 but may train better later.

**What our run actually showed** (README §4): over 5M tokens the order stayed the same as at step 0. Partition r = 0
trained best, then r = 0.25, 0.5 and 1.0, so the "recovers faster" advantage of r = 0.5 did not appear at our short
budget. Redraw did keep more experts in use: 116 of 320 sat nearly idle with r = 0, against 45 with r = 0.5.

---

## 3. Sort-based dropless dispatch

**The problem.** Each token goes to different experts. A GPU is fast only when it runs one big matrix
multiply per expert, so we need **all tokens for expert 0 together, all for expert 1 together, …**,
then put the results back in the original order.

Example: 6 tokens, 3 experts, top-1:

| token | t0 | t1 | t2 | t3 | t4 | t5 |
|---|---|---|---|---|---|---|
| chosen expert | 2 | 0 | 2 | 1 | 0 | 2 |

Steps:
1. **Sort by expert id** (`argsort`): order = t1, t4, t3, t0, t2, t5.
2. **Count per expert** (`bincount`): [2, 1, 3].
3. **Split** the sorted tokens into chunks of 2, 1, 3: E0 gets [t1, t4], E1 gets [t3], E2 gets [t0, t2, t5].
4. **Run each expert** on its chunk, one matrix multiply each, whatever the chunk length.
5. **Multiply** each output by its router weight and **scatter-add** it back to its original token
   position (`index_add_`). With top-k > 1, each token appears k times in the list and its k weighted
   outputs add up in the same slot.

**"Dropless"** means no expert has a size limit. Compare with a **capacity** limit
(lesson §11): capacity = tokens × k / experts × capacity factor = 6 × 1 / 3 × 1.0 = **2**.
E2 received 3, so t5 would be **dropped** (it skips the FFN and only goes through the residual connection).
In dropless dispatch every token is processed. The chunks just have different lengths. The lesson
says current models (DeepSeek-V3, OLMoE, Qwen3) are dropless because dropping costs quality.

---

## 4. "Aux-free bias balancing plus a small per-sequence aux loss"

This is the current recipe (DeepSeek-V3, GLM-4.5, Kimi K2), and it is what our main MoE run uses:

- **Bias (γ = 1e-3)** does the main balancing over the **whole batch** (§1d). It never touches the
  gradient.
- **A tiny aux loss (α = 1e-4) computed per sequence**: the same $N \sum f_i P_i$ formula as §1c, but
  $f$ and $P$ are counted inside **one sequence** (512 tokens), then averaged over the sequences in the batch.
  It only exists to stop an extreme case: one sequence sending all its tokens to one expert, which the
  batch-wide bias would not notice.

Why so small (1e-4, 100× smaller than Switch's 0.01)? Lesson §14: balancing each small group strictly
forces every sequence to use every expert, which stops experts from specializing (e.g. on stories with
dialogue vs. narration). So the per-sequence term should be a safety net, not the main balancer.

---

## 5. The capacity experiment (notebook §11): what was run, what was predicted

**No training happens here.** We take the finished main MoE (after its 20M tokens) and only *evaluate* it on the
validation set 6 times: once with no limit ("dropless") and once with each capacity factor (CF) 0.5, 1.0, 1.25, 1.5
and 2.0.

### "1,024 windows"

The validation set is 1,024 pieces ("windows") of 512 tokens cut from the validation stories, 524,288 tokens in total.
They go through the model in **16 batches of 64 windows** (64 × 512 = 32,768 tokens per batch). Every validation loss
in the notebook uses all 1,024 windows, so all the numbers are comparable.

### Capacity and dropping

In each batch, every token picks 4 experts, so each layer makes 32,768 × 4 = 131,072 picks ("assignments"). Shared
evenly over 32 experts, that's **4,096 per expert**. A capacity limit gives each expert a fixed number of slots:

| CF | slots per expert per batch |
|--:|--:|
| 0.5 | 2,048 |
| 1.0 | 4,096 |
| 2.0 | 8,192 |

When an expert is full, the next tokens that chose it are **dropped**: that expert's output is not added for them. The
token still gets its other experts and the residual connection, so it doesn't vanish, it just gets a weaker FFN in
that layer. "Dropless" means no limit, which is how the model was trained.

### "Dropped, predicted" (from the dropless counts)

Before running with a limit, we can **predict** how many picks will be dropped:

1. Run the **first batch only**, with no limit, and count how many picks each expert gets in each layer. These are
   the **dropless counts**.
2. With capacity C, an expert that got more than C loses the extra: overflow = max(0, count − C).
3. Predicted drop share = total overflow ÷ total picks, averaged over the 10 layers.

Tiny example, 4 experts, 8 picks, CF 1.0 → C = 2 slots each:

| expert | 0 | 1 | 2 | 3 |
|---|--:|--:|--:|--:|
| dropless count | 4 | 2 | 1 | 1 |
| overflow over C = 2 | **2** | 0 | 0 | 0 |

Predicted drop share = 2 / 8 = **25%**.

### "Dropped, measured"

Then we actually run **all 16 batches** with the limit switched on and count the picks that really got dropped. The
measured value differs a little from the prediction for two reasons. First, the other 15 batches have slightly
different loads than the first one. Second, once tokens are dropped in one layer, the inputs to the next layers change
a little, so the counts there change too.

| CF | dropped, measured | dropped, predicted | val loss | worse than dropless by |
|--:|--:|--:|--:|--:|
| 0.5 | 50.95% | 50.45% | 3.6990 | +1.886 |
| 1.0 | 7.11% | 7.86% | 1.8522 | +0.039 |
| 1.25 | 1.88% | 1.82% | 1.8177 | +0.005 |
| 2.0 | 0.28% | 0.05% | 1.8140 | +0.001 |
| dropless | 0 | 0 | 1.8130 | – |

### What was predicted and what was measured

* **Predicted:** only the drop share, from one batch of dropless counts.
* **Measured:** the drop share *and* the validation loss, on all 1,024 windows. **The loss was never predicted.** The
  right plot just shows the measured loss for each CF.

### The two plots

* **Left plot:** drop share against CF, measured (dots) and predicted (crosses). They nearly overlap, which shows the
  drop share can be worked out from the routing counts alone.
* **Right plot:** validation loss against CF, with the dropless loss as a dashed line. With room to spare (CF ≥ 1.25)
  dropping costs almost nothing. At CF 1.0 it costs 0.039, which is more than the MoE's whole gap to the dense model.
  At CF 0.5, half of all picks are dropped and the model breaks.

**Why the lesson cares:** older MoEs (Switch, GShard) used capacity limits, so that every expert did the same amount
of work. Current ones (DeepSeek-V3, Qwen3, OLMoE) are dropless, because dropping costs quality, as the right plot
shows.
