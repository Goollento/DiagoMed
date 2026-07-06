# DiagoMed — Biosignal Denoising Demo

A small, self-contained proof-of-concept that demonstrates how a noisy
biological signal (such as a pulse-oximetry waveform or a slow temperature
trace) can be recovered using a layered, classical digital-filtering
pipeline.

This repository is a personal pet project exploring classical biosignal
denoising techniques.

---

## What the script does

`denoise_demo.py`:

1. **Generates a clean synthetic signal** — a 1.2 Hz sine wave
   (~72 bpm) superimposed on a slow 0.05 Hz baseline drift, sampled at
   100 Hz for 10 seconds.
2. **Corrupts the signal with a layered, realistic noise model:**
   - White Gaussian noise
   - 1/f ("pink") noise, standing in for slow electrode-drift noise
   - 50 Hz mains interference
   - Short EMG-like high-frequency bursts (muscle artefact / shivering)
   - Motion-artefact pulses (electrode pops / contact loss)
3. **Denoises** the signal in three stages, each targeting a different
   kind of noise:
   1. **Hampel filter** — a robust, nonlinear sliding-window filter
      (median + MAD) that strips isolated spikes and short artefact
      runs (motion artefacts) without smoothing the rest of the
      waveform.
   2. **4th-order Butterworth low-pass filter** (cut-off 4 Hz), applied
      via `scipy.signal.filtfilt` for zero-phase distortion, to remove
      the bulk of the broadband/high-frequency noise (50 Hz mains,
      white Gaussian noise).
   3. **Wavelet shrinkage** (soft threshold, shallow decomposition,
      `sym8`) — runs last, on the already-smoothed signal, to mop up
      any residual short, wideband motion/EMG bursts that survive the
      first two stages. Its noise threshold is estimated from the
      signal *before* Butterworth (which still has its high-frequency
      content intact), then applied to shrink the coefficients of the
      signal *after* Butterworth — decoupling "how noisy was this
      signal" from "what do I shrink" avoids the threshold collapsing
      to near-zero once Butterworth has already removed most of the
      high-frequency energy.
4. **Plots** the clean, noisy, and each denoising stage on a single
   figure and saves the result to `denoise_demo.png`.
5. Prints the **SNR (dB)** against the known ground-truth clean signal
   at every stage of the pipeline, so the contribution of each step is
   visible on its own.

The pipeline is encapsulated in four reusable functions:

```python
hampel_filter(signal, window_size=5, n_sigmas=2.5)                                  -> np.ndarray
denoise_signal(noisy_signal, fs=100, cutoff_hz=4.0, order=4)                        -> np.ndarray
wavelet_denoise(signal, noise_reference=None, wavelet="sym8", level=4, mode="soft") -> np.ndarray
compute_snr_db(reference, test_signal)                                              -> float
```

so the pipeline can be reused, reordered, or applied stage-by-stage on
real recordings.

---

## Denoising pipeline

```
Noisy signal
     │
     ▼
┌─────────────────┐     ┌────────────────────┐     ┌───────────────────┐
│  Hampel filter   │ ──▶ │ Butterworth low-pass│ ──▶ │ Wavelet shrinkage │ ──▶ SNR (dB)
│ removes outliers │     │ removes bulk noise  │     │ mops up residual  │
└─────────────────┘     │ (zero-phase)         │     │ motion/EMG bursts │
                         └────────────────────┘     └───────────────────┘
```

Each stage targets a distinct failure mode:

| Stage | Noise it targets | Why |
|---|---|---|
| Hampel filter | Isolated spikes, motion-artefact pulses | Robust to outliers; ignores the rest of the waveform |
| Butterworth low-pass | Gaussian/pink broadband noise, 50 Hz mains | Flat passband, zero phase shift via `filtfilt` |
| Wavelet shrinkage | Residual wideband/EMG bursts | Time-frequency localization catches whatever short bursts survive the first two stages; easier to tune on an already-smoothed signal |

---

## Why this matters for medical diagnostics

Real-world physiological signals — PPG, ECG, skin temperature, EEG —
are routinely contaminated by:

- Broadband electronic noise from the acquisition hardware,
- Slow, correlated (1/f-like) drift from electrodes and skin contact,
- Mains interference at 50 Hz (Europe) or 60 Hz (US),
- Motion-artefact spikes and pops from patient movement,
- Short muscle (EMG) bursts from shivering or tension,
- Slow baseline wander from breathing, perspiration, electrode drift.

Reliably recovering the underlying physiological waveform is the
prerequisite for any downstream diagnostic algorithm
(heart-rate detection, SpO₂ estimation, arrhythmia screening, etc.).
This demo is intentionally minimal — one synthetic signal, three
classical/complementary filtering stages — but the structure mirrors
what a production denoising pipeline looks like: cheap outlier removal
first, then a more targeted decomposition step, then a broadband
smoothing pass, with a quantitative check (SNR) at every stage.

---

## Requirements

- Python **3.9+**
- `numpy`
- `matplotlib`
- `scipy`
- `PyWavelets`

Install everything in one command:

```bash
pip install numpy matplotlib scipy PyWavelets
```

---

## How to run

From the project root:

```bash
python denoise_demo.py
```

The script will:

- Print the SNR (dB) at each pipeline stage to the console,
- Open an interactive Matplotlib window with all five traces,
- Save the figure to `denoise_demo.png` in the current directory.

Expected console output (values are deterministic — fixed RNG seed):

```
SNR of noisy signal:                              3.49 dB
SNR after Hampel filter:                           4.08 dB
SNR after Hampel + Butterworth:                     8.10 dB
SNR after Hampel + Butterworth + wavelet:            8.54 dB
Total improvement:                               +5.05 dB
```

---

## Result at a glance

The generated figure shows five stacked panels:

```
┌─────────────────────────────────────────────────────────┐
│  Clean signal        (smooth sine + slow drift)         │
├─────────────────────────────────────────────────────────┤
│  Noisy signal        (Gaussian + pink + 50 Hz +         │
│                        EMG bursts + motion artefacts)    │
├─────────────────────────────────────────────────────────┤
│  Stage 1: Hampel      (spikes/artefacts removed)        │
├─────────────────────────────────────────────────────────┤
│  Stage 2: Butterworth (bulk noise removed, zero-phase)  │
├─────────────────────────────────────────────────────────┤
│  Stage 3: Wavelet     (residual bursts mopped up,       │
│                         dashed reference)                │
└─────────────────────────────────────────────────────────┘
```

Each of the last three panels is titled with its running SNR (dB), so
the improvement contributed by each stage is visible directly on the
plot. The bottom panel overlays the fully recovered trace on top of
the original ground truth (dashed) — they should be visually close.

---

## File layout

```
.
├── denoise_demo.py     # main script (signal gen + denoising pipeline + plotting)
├── README.md           # this file
└── denoise_demo.png    # produced on first run
```

---

## Notes on the choice of filters

- **Hampel filter** is applied first because it is robust to the
  motion-artefact pulses and isolated spikes in the noise model — a
  linear filter (Butterworth) would spread a sharp spike's energy
  across nearby samples instead of removing it.
- **Butterworth low-pass** runs second, over the spike-free signal,
  because it's the cheapest way to remove the bulk of the remaining
  broadband noise and the 50 Hz mains tone:
  - It has a **flat passband** — it does not attenuate the in-band
    physiological content.
  - Combined with `filtfilt`, it introduces **zero phase shift**
    (`filtfilt`, not `lfilter`), which preserves the temporal alignment
    of features (peaks, onsets) — this matters for any time-based
    diagnostic measurement. This is safe here because the demo runs as
    offline post-processing, not a real-time/streaming pipeline; a
    real-time system would need a causal filter instead and would
    accept some phase lag.
  - The cut-off (4 Hz) sits comfortably above the dominant signal
    frequency (1.2 Hz) and well below the mains content.
- **Wavelet shrinkage** runs last, over the already-smoothed signal,
  to mop up whatever short, wideband motion/EMG residual survived the
  first two stages. Its **noise threshold**, however, is estimated
  from `after_hampel` — the signal *before* Butterworth — rather than
  from the signal it actually shrinks:
  - By the time Butterworth has run, the finest wavelet detail
    coefficients of the signal are mostly gone (that's the whole point
    of the low-pass filter), so estimating the noise floor from the
    post-Butterworth signal itself yields a threshold too small to
    meaningfully shrink anything.
  - Estimating sigma from the pre-Butterworth signal instead — which
    still carries its full high-frequency noise character — gives an
    honest measurement of "how noisy was this recording", which is
    then used to threshold the coefficients of the post-Butterworth
    signal — "what do I actually shrink". Keeping those two roles
    separate is what lets the wavelet stage meaningfully suppress
    residual burst energy instead of doing almost nothing.
  Two parameters matter most for the shrinkage step itself:
  - **Soft thresholding** (not hard) — hard-thresholding zeroes
    coefficients outright and leaves small discontinuities in the
    reconstructed waveform, visible as jagged edges. Soft shrinks
    coefficients smoothly toward zero instead.
  - **A shallow decomposition depth** (level 3–5, `level=4` here) with
    a **smooth mother wavelet** (`sym8`, `sym4` or `db4` all work) —
    decomposing too deep starts folding the signal's own 1.2 Hz
    carrier into the detail coefficients, so shrinkage attenuates the
    physiological oscillation itself, not just the artefact bursts.

For artefact sources this pipeline still cannot separate cleanly from
the signal (e.g. large, sustained motion overlapping the physiological
band), the natural next step is a **reference-based** method —
adaptive filtering (LMS/RLS) against a motion-reference channel, or
ICA/EMD across multiple recorded channels — since those require
information this single-channel demo does not have.

---

## License

MIT — feel free to reuse for educational or evaluation purposes.

---

*This repository is a personal demonstration/learning project.*
