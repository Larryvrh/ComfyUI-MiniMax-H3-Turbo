# ComfyUI-MiniMax-H3-Turbo

Run [MiniMax-H3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) —
joint **video + synchronized audio** — in as few as **4 sampling steps** instead
of ~20, with the
[MiniMax-H3 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).

Two nodes drop straight into the official H3 workflow (text-to-video and
image-to-video):

| node | what it does |
|---|---|
| **MiniMax-H3 Turbo LoRA** | `MODEL → MODEL`, applies the turbo LoRA |
| **MiniMax-H3 Turbo Sampler** | `→ SAMPLER`, feeds `SamplerCustomAdvanced` |

## Which checkpoint — `v4` (600) or `v1` (850)?

For **most** work, use **`minimax_h3_turbo_v4_step600_ema.safetensors`** (from the
[LoRA repo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)). It's the
strongest checkpoint so far: much better static / small-motion shots, markedly
better micro-detail (faces, fingers, texture), and the over-sharpening / plastic
look of the earlier `v1` (~850) line is fully resolved.

v4 introduced a **static-frame enhancement**. The one trade-off shows up **only at
4 steps with large, fast motion**, where v4 can produce **motion-smear / trailing
ghosting** (actively being fixed). **Using 6–8 steps largely removes it** — and v4
tolerates higher step counts better than v1 (which over-sharpens at high steps +
strength 1.0). For the narrow case of **4 steps *and* heavy motion**, the older
**`v1` ~850** checkpoint can still be friendlier.

```
Using 6–8 steps?        ── yes ──►  v4-600  (recommended)
   │ no (4 steps)
   ▼
Heavy / fast motion?    ── no  ──►  v4-600  (recommended)
   │ yes
   ▼
                                    v1-850  (friendlier at 4-step heavy motion)
```

Still a preview — the two areas still being improved are **audio** and **fast,
intense motion**.

## Install

**Via ComfyUI-Manager** — search "MiniMax-H3 Turbo" and install. **Or manually:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo
```

Then restart ComfyUI. Keep the node updated (Manager, or `git pull`) — it evolves
alongside the weights.

Put the LoRA `.safetensors` into `ComfyUI/models/loras/`. You also need the base
MiniMax-H3 model, VAEs and text encoder from the official release — see the
[MiniMax-H3 tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3).

## Use

Start from the **official MiniMax-H3 workflow** (t2v or i2v) and make two changes:

1. Insert **MiniMax-H3 Turbo LoRA** between the model loader and the sampler
   (`… → Load Diffusion Model → MiniMax-H3 Turbo LoRA → SamplerCustomAdvanced`),
   and pick the turbo `.safetensors`.
2. Feed `SamplerCustomAdvanced` from **MiniMax-H3 Turbo Sampler**, and set the
   scheduler node (`BasicScheduler`) to `simple` at **≥ 4 steps**.

Everything else — conditioning, VAE decode, audio output — stays as in the
official graph, so **both t2v and i2v** work unchanged. A ready-made t2v workflow
is in [`example_workflows/`](example_workflows/minimax_h3_t2v_turbo.json) — drag it
into ComfyUI to see the wiring.

## Steps and strength

- **4 steps is the recommended *minimum*; 4–8 is the useful range.** 6–8 steps
  look noticeably better than 4, so add steps if you can afford them. Past **8
  steps** it stops helping and can start to introduce **over-sharp artifacts** —
  no benefit to going higher, so stay in **4–8**.
- **Keep `strength` at `1.0`.** It's tuned for 1.0 and holds up across the 4–8 step
  range. Only reach for the dial if a *specific* clip misbehaves: **blurry ghosting
  / smear → nudge up** (`~1.05–1.2`), **over-sharp grain → nudge down**
  (`~0.8–0.95`).
- Keep the scheduler on `simple`.

## Base model & `low_vram`

**Base model** — works with any MiniMax-H3 base: full (`bf16`, `int8_convrot`)
**and the pruned/curve variants** (`pruned_int8`, `pruned_fp8`). The node detects a
pruned base automatically and re-injects the LoRA's time-conditioning at run time
(a small `silu(t_emb)` grid ships with the node for this), so **one LoRA file
covers every base**.

**`low_vram`** (node switch) trades sharpness for peak VRAM:

- **off (default)** — applies the LoRA at run time (bypass): sharpest, recommended,
  a little extra peak VRAM.
- **on** — merges the LoRA into the weights: lowest peak VRAM, so smaller GPUs run
  and longer / higher-res clips fit, but the result is **softer on quantized
  (`int8` / `fp8` / pruned) bases** (the tiny update is partly rounded away when
  folded into the quantized weights).

The node streams the base model, so it runs on much smaller GPUs than the ~33 B
size suggests — an 80 GB GPU is only needed for the largest resolutions in `bypass`
mode. If you hit out-of-memory, turn `low_vram` **on** and/or lower the resolution
or frame count.

### Why you need this node (not just a LoRA loader)

The turbo LoRAs are authored against a base whose `adaln_proj` path is at one size;
on a base where that path differs, a **stock loader silently applies only part of the
LoRA** instead of failing loudly.

Measured instance — `minimax_h3_turbo_v4_step600` through `LoraLoaderModelOnly` on a
pruned INT8 base:

- **208 of 259 modules applied (80.3%)**
- **51 skipped, all of them `adaln_proj.linear`**
- one `ERROR lora` line per layer, i.e. 50 per sampling step

```
ERROR lora ... adaln_proj linear.weight shape '[96768, 8]' is invalid
for input of size 260112384
```

AdaLN is exactly the pathway step-distillation is meant to change, so a partially
applied turbo LoRA is not the model that was trained and benchmarked.

**MiniMax-H3 Turbo LoRA does not have this problem**: it detects a pruned base and
re-injects the LoRA's time-conditioning at run time — that is what the `silu(t_emb)`
grid is for.

**How to tell which you got:** after the LoRA loads, search the console for
`adaln_proj`. Repeated `ERROR lora` lines mean a partial load — swap the loader for
this node.

## Why a custom sampler (and how it adapts)

MiniMax-H3 denoises the video and audio streams on two different flow schedules
(video shift 12, audio shift 3). **Recent ComfyUI handles this natively** — its
`ModelSamplingAV` carries the audio latent on the video schedule — so a stock
sampler already produces correct audio there. On **older ComfyUI without that
support**, a stock sampler steps both streams on one schedule and badly over-steps
the audio at 4 steps, so the audio comes out distorted.

This node's sampler **auto-detects which ComfyUI it's on**: on recent builds it
steps as a plain single-schedule sampler (bit-for-bit the stock result); on older
builds it steps each stream on its own clock so audio stays clean at 4 steps. Keep
it in the workflow and it does the right thing across ComfyUI versions — nothing to
change when you update. (On recent ComfyUI a stock `euler` also works; the Turbo
Sampler just keeps existing graphs running unchanged.)

## Notes

- **Resolution / length**: width and height are multiples of 32 (short edge
  typically 768); frame count is at 24 fps and snaps to the model's 17·k+5 grid
  (124 ≈ 5 s). Validated range ~124–362 frames.

## License

Apache-2.0.
