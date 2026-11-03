# Assignment 12: Data Parallelism and ZeRO on 32 Virtual GPUs

Training one weight of a language model takes 16 bytes of memory, so a 30-billion-weight model needs 447 GiB, about
six 80 GB GPUs, before it stores a single activation. The way out is to split the work and the memory over many GPUs,
and the price is communication between them. For this assignment I built 32 virtual GPUs, wrote data parallelism and
ZeRO stages 1, 2 and 3 from scratch on top of them, and measured how memory, communication and computation change.
Each result is predicted by hand first and then measured.

| | |
|---|---|
| Virtual GPUs | 32 operating-system processes started by `torchrun`, each with its own private memory, talking through `torch.distributed` (gloo, PyTorch's CPU backend) |
| Communication | ring collectives written by hand on top of send/receive ([`dist_lib.py`](dist_lib.py)), so every byte each GPU sends is counted |
| ZeRO engine | DDP, ZeRO-1, ZeRO-2 and ZeRO-3 written from scratch, with a byte-exact memory ledger per GPU ([`zero_sim.py`](zero_sim.py)) |
| Model | the Llama-style decoder from Assignments 10 and 11 ([`model.py`](model.py)), 4 layers, width 256, **5.3M weights**, split into 6 *units* (embedding, 4 blocks, head) |
| Data | TinyStories, 60,000 stories, 13.3M tokens, vocabulary reduced to the 4,096 most common GPT-2 tokens (98.8% of the text kept) |
| Hardware | Colab A100 runtime, used for its 12 CPU cores and 167 GiB of RAM. The whole notebook runs in 18 minutes |

Notebook: [`session12_zero.ipynb`](session12_zero.ipynb), executed with all outputs saved. Every number below comes
from that run.

## Results

| Task | Result |
|---|---|
| 1. Create 32 virtual GPUs | 32 ranks, 32 separate processes; all three collectives exact on hand-checked numbers |
| 2. A demo model running on them | 150 training steps under ZeRO-3 on 32 GPUs: held-out loss **4.349**, against **4.346** on one process |
| 3. Simulate ZeRO-1, 2 and 3 | All four arrangements give **bit-identical weights**, and match one GPU to 1.3e-14 |
| 4a. How memory changes (per GPU, W = 32) | DDP **16** → ZeRO-1 **4.375** → ZeRO-2 **2.4375** → ZeRO-3 **0.5** bytes per weight, exactly the formulas |
| 4b. How communication changes | **1.94 P** for DDP, ZeRO-1 and ZeRO-2; **2.91 P** for ZeRO-3 (P = one copy of the weights) |
| 4c. How computation changes | Forward and backward unchanged; the optimizer runs on 1/32 of the weights and gets **75× faster** |

---

## Words used below

| Term | Meaning |
|---|---|
| rank, world size W | each GPU has a number (its rank), from 0 to W−1; W is how many GPUs take part |
| collective | an operation every GPU calls at the same time, each holding a piece of the data |
| P | the size of one full copy of the weights in bytes (bf16: 2 bytes × number of weights) |
| replicated / sharded | stored in full on every GPU / cut into W slices, one slice per GPU |
| unit | the piece of the model ZeRO handles at once: here the embedding, one transformer block, or the head |
| fp32 master copy | a 4-byte copy of every weight kept by the optimizer, because tiny updates get lost when added to a 2-byte weight |
| activation checkpointing | keeping fewer intermediate results in the forward pass and recomputing them in backward |

---

## Task 1: Thirty-two virtual GPUs

A real multi-GPU job is W copies of one program, each with memory the others cannot read. Data moves between them
only through explicit messages. So each virtual GPU here is one **process**:

| Option | Own memory? | Real messages? | Used? |
|---|---|---|---|
| 32 Python threads | no, threads share memory | no | no |
| a Python list of 32 "GPUs" | no | no, a `for` loop plays the network | no |
| 32 ranks on the one Colab GPU | yes | NCCL refuses two ranks on one GPU | not possible |
| **32 processes with `torch.distributed`** | **yes** | **yes**, over local sockets | **yes** |

The code is the same as real multi-GPU code. Moving it to real GPUs means changing the backend from `gloo` to `nccl`.

### The three collectives

| Collective | Each GPU starts with | Each GPU ends with |
|---|---|---|
| all-reduce | its own vector | the sum of all vectors |
| reduce-scatter | its own vector | one slice of the sum (GPU r gets slice r) |
| all-gather | one slice | all the slices |

A reduce-scatter followed by an all-gather gives exactly an all-reduce. Everything ZeRO does is built on that fact.

**Check on numbers I could predict by hand.** GPU r holds the values 100r + c for c = 0 … 31, so slice c of the sum
is 49,600 + 32c. Measured: 49,600, 49,632, 49,664, … on gloo's all-reduce, exactly 49,600 + 32r on GPU r after my
reduce-scatter, and reduce-scatter + all-gather equal to all-reduce with difference **0.0**.

### The ring all-reduce, and why it scales

The GPUs sit in a ring and each one only sends to its right neighbour. Every vector is cut into W chunks. In each
round a GPU passes one chunk on and adds the chunk it receives to its own, so the partial sums travel round the ring
and grow by one contribution per round. After W−1 rounds every GPU holds one complete chunk (that is the
reduce-scatter), and W−1 more rounds pass the complete chunks around (the all-gather).

$$\text{bytes sent per GPU} = 2\,(W-1) \cdot \frac{n}{W} \;\approx\; 2n$$

- $n$ is the size of the vector in bytes, $W$ the number of GPUs.
- $2(W-1)$ is the number of sends, and each send carries $n/W$ bytes.

This **does not grow with W**: adding GPUs adds rounds, not bytes per GPU. For a 128 KiB message on 32 GPUs I counted
253,952 bytes sent per GPU, exactly $2 \times \frac{31}{32}$ of the message. This $2\frac{W-1}{W}$ is the lesson's
"2P".

The time of one all-reduce has two parts, a fixed start-up cost per send (latency) and the bytes divided by the
link speed (bandwidth):

$$t(n) = 2\,(W-1)\,\alpha \;+\; 2\,\frac{W-1}{W}\,\frac{n}{\beta}$$

- $\alpha$: the latency of one send. Fitted: **653 µs**.
- $\beta$: the bandwidth of one link. Fitted: **0.35 GB/s**.

![all-reduce time vs message size](ring_allreduce_time.png)

*X-axis: message size per GPU (log scale). Y-axis: time of one all-reduce (log scale). Blue: my ring. Orange: gloo's
built-in all-reduce. Dashed: the fit above.*

The time is flat at about 40 ms up to about 512 KiB, because latency dominates, and only grows with size above about
1 MiB. These numbers are poor (32 processes take turns on 12 cores, so a GPU often waits for its neighbour to be
scheduled), but the shape is the real one. It is also the reason real systems send gradients in **buckets**:
sending this model's 31 weight tensors one by one would cost 31 × 41 ms = 1.3 s of latency alone, against 0.25 s
for 6 unit-sized buckets.

---

## Task 2: The demo model

The decoder from Assignments 10 and 11 (RMSNorm, rotary position embeddings, SwiGLU MLP), shrunk so that 32 copies
fit on 12 CPU cores: 4 layers, width 256, 5,310,720 weights.

ZeRO does not handle the model as one block or weight by weight, but in **units**: here the embedding, each of the 4
blocks, and the final norm with the head. Each unit's weights are laid out as one flat vector and cut into W equal
slices. GPU r owns slice r of every unit, which at W = 32 is 165,960 weights.

| Unit | Weights | Share of the model |
|---|---:|---:|
| embedding | 1,048,576 | 19.7% |
| each of the 4 blocks | 803,328 | 15.1% |
| final norm + head | 1,048,832 | 19.7% |

Every step uses the same 64 sequences of 128 tokens whatever the number of GPUs. Each GPU takes its share, so a run
on 1 GPU and a run on 32 see exactly the same data. That is what makes the exact comparisons below possible.

---

## Task 3: ZeRO-1, ZeRO-2 and ZeRO-3

The starting point is **data parallelism (DDP)**. Every GPU holds the whole model and all 16 bytes per weight, and
runs forward and backward on its own 2 sequences. The GPUs then **all-reduce** the gradients and divide by W, and
every GPU runs the same Adam update. The average of 32 per-GPU gradients is exactly the gradient of all 64 sequences,
so DDP is the same training run as one GPU with a 32 times larger batch.

The 16 bytes per weight are:

| Stored per weight | Format | Bytes | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---:|---|---|---|---|
| weight used for compute | bf16 | 2 | full | full | full | **slice** |
| gradient | bf16 | 2 | full | full | **slice** | slice |
| fp32 master copy | fp32 | 4 | full | **slice** | slice | slice |
| Adam's m and v | fp32 | 8 | full | **slice** | slice | slice |
| **bytes per weight on each GPU** | | | $16$ | $4 + \frac{12}{W}$ | $2 + \frac{14}{W}$ | $\frac{16}{W}$ |

### ZeRO-1: split the optimizer state

In DDP all 32 GPUs run the same Adam update on the same numbers. ZeRO-1 makes GPU r responsible for **slice r** of
the weights only, so it keeps the master copy and Adam's m and v for that slice only. Each step:

1. forward and backward give every GPU a full gradient, as in DDP;
2. **reduce-scatter** the gradients: GPU r receives the summed gradient of slice r;
3. GPU r runs Adam on its slice and writes the new weights into its copy;
4. **all-gather** the weights, so every GPU has the full updated model again.

**It costs no extra communication.** DDP's all-reduce already *is* a reduce-scatter plus an all-gather. ZeRO-1 does
the same two steps, but gathers the new weights instead of the summed gradients.

### ZeRO-2: split the gradients too

After the reduce-scatter, GPU r only ever uses slice r of the gradient. ZeRO-2 throws the rest away as early as
possible. Backward produces gradients one unit at a time, from the head back to the embedding. As soon as a unit is
complete it is reduce-scattered and its full gradient freed. The full gradient of one unit exists for a moment, never
the whole model's. Communication is unchanged: the reduce-scatter only moves into the backward pass.

### ZeRO-3: split the weights too

Each GPU stores only its slice of the weights. A unit's full weights exist only while that unit computes:

| Moment | What happens |
|---|---|
| forward, each unit | all-gather its weights, run it, free them |
| backward, each unit | all-gather its weights **again**, compute its gradients, reduce-scatter them, free everything |
| optimizer | update the slice; no all-gather needed, the next forward gathers anyway |

That is one extra all-gather per step, so 3 instead of 2 copies of the weights are sent: the lesson's "3P".

**How I implemented the "free and gather again".** If PyTorch kept a unit's normal computation graph, the graph would
hold on to the gathered weights and they could not be freed. So each unit runs without recording a graph and keeps
only its input; in backward it gathers the weights again and **recomputes** the unit. That is activation
checkpointing: about 33% more compute and much smaller activations. PyTorch's FSDP avoids the recompute with a
different trick, but memory and bytes are the same. To keep the comparison fair, I also ran all the other stages
with checkpointing.

### All four compute exactly the same thing

After 4 training steps in fp64 (8-byte numbers, so that rounding cannot hide a bug):

| | Largest weight difference |
|---|---:|
| any ZeRO stage vs DDP | **0** (bit-identical) |
| any stage vs one GPU on all 64 sequences | 1.3e-14 |
| for comparison: how far training moved the weights | 3.0e-3 |

The small difference from one GPU comes from batching (one matrix multiply over 64 sequences against 32 over 2),
not from ZeRO. **ZeRO changes where the numbers are stored and who computes which update, never what is computed.**

---

## Task 4: How memory, communication and computation change

### Memory per GPU (W = 32)

| | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---:|---:|---:|---:|
| bytes per weight, predicted | 16 | 4.375 | 2.4375 | 0.5 |
| bytes per weight, measured | **16.000** | **4.375** | **2.4375** | **0.500** |
| MiB per GPU between steps | 81.04 | 22.16 | 12.35 | 2.53 |
| MiB per GPU at the peak inside a step | 81.04 | 22.16 | 16.35 | 6.53 |
| activations kept for backward (MiB) | 29.0 | 29.0 | 29.0 | 5.25 |

- **The formulas hold exactly.** Under DDP, 32 GPUs hold 2,593 MiB to store 81 MiB of information.
- **The peak is higher than the between-steps number** for ZeRO-2 and ZeRO-3: while a unit is being reduced or
  gathered, its full buffers exist for a moment. That extra is set by **the largest unit, not the model size**, which
  is why bucket size (DeepSpeed) and wrapping granularity (FSDP) are tuning settings. In this small model the head unit
  is 20% of all weights, so the extra is large; in a 30B model with 60 layers one layer is under 2%.
- **ZeRO does not shrink activations.** They are the same 29 MiB in every stage. Only checkpointing reduces them
  (5.25 MiB in every stage), which is why ZeRO-3 is almost always used together with it.

To check against the lesson's table with real 2-byte numbers, I ran all four stages with genuine bf16 tensors on 8
GPUs. The measured bytes per weight were **16.000, 5.500, 3.750, 2.000**, exactly the lesson's values.

### Communication per GPU per step

| | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| what is sent | all-reduce of gradients | reduce-scatter of gradients + all-gather of weights | same as ZeRO-1, during backward | all-gather in forward + all-gather in backward + reduce-scatter |
| bytes sent, in P | **1.9375** | **1.9375** | **1.9375** | **2.9063** |

ZeRO-1 and ZeRO-2 add only 496 bytes per step, for summing the gradient norm across GPUs (each GPU only holds a slice
of the gradient).

![memory, communication and time of the four stages](stages_memory_comm_time.png)

*Left: memory per GPU, split by what is stored (black marks: peak). Middle: bytes sent per step, split by purpose.
Right: time per step; left bars as implemented, hatched bars with every stage using checkpointing.*

### Computation per GPU (ms per step, W = 32)

| | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---:|---:|---:|---:|
| weights each GPU's optimizer updates | 5,310,720 | 165,960 | 165,960 | 165,960 |
| optimizer compute | 325 | 4.3 | 4.2 | 3.3 |
| forward + backward compute | 499 | 476 | 444 | 564 |
| communication (including waiting) | 450 | 492 | 566 | 843 |
| whole step | 1,275 | 973 | 1,014 | 1,411 |

- **ZeRO does not reduce the forward and backward work.** Every GPU still runs the whole model on its own sequences.
  ZeRO-3 is higher only because of its recomputation.
- **The optimizer is where ZeRO saves compute:** 75 times less time for 32 times fewer weights. The extra factor comes
  from the CPU cache: a 166k-weight slice fits in it, while 5.3M weights × 5 tensors in 32 processes do not. Here that
  made ZeRO-1's whole step 24% faster than DDP's.
- **ZeRO-3's communication takes 1.79 times DDP's for 1.5 times the bytes.** It makes 19 collectives per step instead
  of 6, and each pays the latency.

### Changing the number of GPUs

The same 64 sequences per step, split over 1, 2, 4, 8, 16 and 32 GPUs:

![world-size sweep](world_size_sweep.png)

*X-axis of all three: number of GPUs (log scale). Left: bytes per weight on each GPU (dots measured, lines formula,
crosses the peak inside a step). Middle: bytes sent per GPU per step, as a multiple of P. DDP, ZeRO-1 and ZeRO-2 send
identical bytes, so their curves lie on top of each other (big circle, diamond, small dot). Right: time per GPU.*

| Number of GPUs | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| ZeRO-3 bytes per weight | 16 | 8 | 4 | 2 | 1 | 0.5 |
| ZeRO-1 bytes per weight | 16 | 10 | 7 | 5.5 | 4.75 | 4.375 |
| bytes sent by DDP / ZeRO-1 / ZeRO-2, in P | 0 | 1.0 | 1.5 | 1.75 | 1.875 | 1.9375 |
| bytes sent by ZeRO-3, in P | 0 | 1.5 | 2.25 | 2.625 | 2.8125 | 2.906 |
| forward + backward per GPU, DDP (ms) | 3,401 | 1,737 | 887 | 533 | 497 | 514 |
| optimizer per GPU, DDP (ms) | 49 | 54 | 72 | 95 | 168 | 310 |
| optimizer per GPU, ZeRO-1 (ms) | 50 | 25 | 13 | 10 | 6.7 | 4.4 |

- **Memory:** ZeRO-1 and ZeRO-2 level off at 4 and 2 bytes per weight (the weights and gradients they still keep in
  full); only ZeRO-3 keeps halving.
- **Bytes per GPU level off** at 2P and 3P. More GPUs do not mean more traffic per GPU.
- **Compute per GPU falls like 1/W** up to 8 GPUs, then stops because the machine has only 12 cores. DDP's optimizer
  even gets slower with more GPUs (the same full update in more processes competing for memory), while ZeRO's keeps
  falling.
- **Communication overtakes compute** as W grows: ZeRO-3 spends 32 ms communicating at 2 GPUs and 856 ms at 32. That
  is the lesson's point about faster GPUs, reproduced here by giving each GPU less work.

---

## Extra: training for real

![training under ZeRO-3 vs one process](training_zero3_vs_single.png)

*Training loss per step (lines) and held-out loss every 25 steps (markers), ZeRO-3 on 32 GPUs against one process.*

| Step | 1 | 25 | 75 | 150 |
|---|---:|---:|---:|---:|
| held-out loss, ZeRO-3 on 32 GPUs | 8.276 | 5.860 | 4.846 | **4.349** |
| held-out loss, one process | 8.276 | 5.860 | 4.846 | **4.346** |

No GPU ever held more than 2.5 MiB of training state between steps. The two runs agree to 6e-7 up to step 25, then
differ by up to 2.8e-2 in training loss. I had predicted about 1e-3. In fp32 the two runs do not do exactly the same
arithmetic, and training turns tiny rounding differences into slightly different paths. The fp64 test above shows the
method itself is exact.

---

## Extra: three deliberate bugs

![bugs](bugs_drift_and_loss.png)

*Left: largest weight difference between GPU copies (log scale). The correct runs stay at exactly 0. Right: training
loss under each bug.*

| Bug | What happens | Measured |
|---|---|---|
| DDP without averaging the gradients | each GPU applies its own gradient, so the copies drift apart | drift **6.0e-4 after step 1, exactly 2× the learning rate**, 5.6e-2 after 20 steps. The loss looks only a little worse (6.04 vs 5.90), so a loss curve would not reveal it |
| ZeRO-1 without the weight all-gather | each GPU updates its slice but never receives the others | loss stalls at **7.96 vs 5.90**: each GPU trains a model that is 7/8 frozen |
| Gradients summed instead of averaged (8× too large), no clipping | Adam divides by the gradient's own size, so a constant factor cancels | loss changes by only **4.2e-4** |
| Same bug, with gradient clipping at 1.0 | clipping looks at the raw size, so it now clips on every step | loss changes by **1.5e-2** (clipped on 20 of 20 steps instead of 9) |

The third bug is the surprise: an 8 times error in the gradient barely matters under Adam. I predicted about 1e-5;
the measured 4.2e-4 is larger because for weights with almost no gradient (rare tokens) Adam's tiny ε is no longer
negligible.

---

## Extra: one real node, and the course's capstone model

The 32-GPU setup behaves like **one machine with 32 GPUs** joined by one kind of wire. Real machines (nodes) have
**8 GPUs** joined by NVLink (450 to 900 GB/s), and nodes are joined by InfiniBand (about 50 GB/s). A ring that leaves
the node runs at InfiniBand speed. On Colab's 12 cores, 8 GPUs can each get **their own CPU core**, which removes the
"waiting for a core" effect:

![one node vs 32 ranks](single_node_8_vs_32.png)

*Share of each step spent on each activity, with every stage checkpointed. Left bar: 32 GPUs on 12 cores. Hatched
bar: 8 GPUs, one core each.*

| One node, 8 GPUs, one core each | Measured |
|---|---|
| forward compute | 181 to 200 ms in every stage |
| compute speed | **49 to 52 GFLOP/s per core in every stage**: ZeRO changes memory and messages, not arithmetic |
| communication / compute | **29 to 39%**, against 48 to 152% for 32 GPUs on the same 12 cores |

**What this means for the capstone.** The course plans to train one model of about 30B weights together. Applying the
measured rules to real B200 GPUs (180 GB each) gives the following. This is a projection, assuming 40% MFU, no overlap
of communication with compute, and activations not counted:

| Setup | Fits? | Compute per 1M-token step | Communication share | Days for 300B tokens |
|---|---|---:|---:|---:|
| 1 × B200 | **no**: 447 GiB needed, 168 GiB available | 200 s | none | 694 |
| 1 node, 8 × B200 | yes, every ZeRO stage (ZeRO-3: 56 GiB per GPU) | 25 s | **0.5 to 0.7%** (NVLink) | **87** |
| 4 nodes, 32 × B200 | yes | 6.3 s | 27 to 36% (InfiniBand) | 30 to 34 |

One node holds the model easily and its communication is almost free; the limit is time. For a single node I would
choose **ZeRO-2**: on 180 GB cards it fits with about 63 GiB left for activations, and it sends a third less data than
ZeRO-3.

---

## What I learned

1. **ZeRO is bookkeeping, not mathematics.** Four arrangements, bit-identical weights. Only who stores which slice,
   and who updates it, changes.
2. **ZeRO-1 and ZeRO-2 are free in communication**, because an all-reduce already is a reduce-scatter plus an
   all-gather; they just keep the slice in between. ZeRO-3 pays one more all-gather (2P becomes 3P).
3. **Bytes per GPU do not grow with the number of GPUs.** Latency does, and so does the share of time spent
   communicating, because each GPU has less work to hide it behind.
4. **Memory has two numbers.** Between steps it follows the formulas exactly. At the peak one unit's full buffers come
   on top, and activations are not touched by ZeRO at all.
5. **ZeRO does not cut forward and backward work per GPU.** It cuts the optimizer's share, which on this CPU was a
   quarter of the DDP step.

## Limits of this work

- **Virtual GPUs are CPU processes.** Memory and bytes are exact, but times are not GPU times. With 32 processes on
  12 cores, "communication" includes waiting for a core. Only ratios and trends mean something; the 8-GPU node with one
  core per GPU gives the most reliable timings.
- **Precision.** The exact comparisons run in fp64 and the timings in fp32; memory is converted to the lesson's bytes
  from exact element counts. Only the 8-GPU bf16 run stores real 2-byte numbers.
- **ZeRO-3 recomputes** each unit in backward, unlike PyTorch's FSDP. Memory and bytes are the same; compute is about
  33% higher. All other stages were also run with checkpointing to separate the two effects.
- **Not simulated:** overlapping communication with compute, CPU or NVMe offload, weight decay. The single-node versus
  multi-node comparison is a calculation from stated bandwidths, not a measurement.
- **Small model:** the largest unit is 20% of the model, so the peaks inside a step look large.
- **Single seed and short runs:** 4 steps for the exact comparisons, 2 to 7 timed steps, 150 training steps and 20
  steps per bug.

## How to run

1. Open [`session12_zero.ipynb`](session12_zero.ipynb) in Colab (or VS Code connected to a Colab kernel). Choose a
   runtime with many CPU cores and plenty of RAM. The A100 runtime (12 cores, 167 GiB) runs everything in about 18
   minutes; the GPU itself is not used.
2. Run all cells. The notebook installs `tiktoken`, writes the four source files, downloads one TinyStories file (no
   token needed) and starts every experiment with `torchrun`.
3. The last cell releases the Colab runtime. In VS Code, save the notebook afterwards to keep the outputs.

Files: the notebook, `dist_lib.py` (ring collectives with byte counting), `model.py` (the model as 6 units),
`zero_sim.py` (DDP and ZeRO-1/2/3), `collectives_demo.py` (the virtual-GPU check, collectives and ring benchmark), and
the plots extracted from the notebook. The `.py` files are identical to the notebook's `%%writefile` cells.
