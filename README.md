# DiagoMed — Biosignal Denoising Demo

A small, self-contained proof-of-concept that demonstrates how a noisy
biological signal (such as a pulse-oximetry waveform or a slow temperature
trace) can be recovered using a layered, classical digital-filtering
pipeline.

This repository was created as a portfolio piece for **inBiome**.

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
   2. **Wavelet shrinkage** (VisuShrink, soft threshold, `db6`) —
      isolates short, wideband bursts (residual motion/EMG artefact)
      in the time-frequency domain — the one thing a fixed window
      (Hampel) and a fixed cutoff (Butterworth) both miss on their own.
   3. **4th-order Butterworth low-pass filter** (cut-off 4 Hz), applied
      via `scipy.signal.filtfilt` for zero-phase distortion, to remove
      the remaining broadband/high-frequency noise.
4. **Plots** the clean, noisy, and each denoising stage on a single
   figure and saves the result to `denoise_demo.png`.
5. Prints the **SNR (dB)** against the known ground-truth clean signal
   at every stage of the pipeline, so the contribution of each step is
   visible on its own.

The pipeline is encapsulated in four reusable functions:

```python
hampel_filter(signal, window_size=5, n_sigmas=2.5)        -> np.ndarray
wavelet_denoise(signal, wavelet="db6", mode="soft")       -> np.ndarray
denoise_signal(noisy_signal, fs=100, cutoff_hz=4.0, order=4) -> np.ndarray
compute_snr_db(reference, test_signal)                    -> float
```

so the pipeline can be reused, reordered, or applied stage-by-stage on
real recordings.

---

## Denoising pipeline

```
Noisy signal
     │
     ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Hampel filter   │ ──▶ │ Wavelet shrinkage │ ──▶ │ Butterworth low-  │ ──▶ SNR (dB)
│ removes outliers │     │ isolates residual │     │ pass (zero-phase) │
└─────────────────┘     │ motion/EMG bursts │     └───────────────────┘
                         └──────────────────┘
```

Each stage targets a distinct failure mode:

| Stage | Noise it targets | Why |
|---|---|---|
| Hampel filter | Isolated spikes, motion-artefact pulses | Robust to outliers; ignores the rest of the waveform |
| Wavelet shrinkage | Residual wideband/EMG bursts | Time-frequency localization catches short bursts a fixed window or fixed cutoff would miss |
| Butterworth low-pass | Gaussian/pink broadband noise, 50 Hz mains | Flat passband, zero phase shift via `filtfilt` |

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
SNR of noisy signal:                           3.49 dB
SNR after Hampel filter:                        4.08 dB
SNR after Hampel + wavelet:                      7.30 dB
SNR after Hampel + wavelet + Butterworth:         8.01 dB
Total improvement:                            +4.52 dB
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
│  Stage 1: Hampel     (spikes/artefacts removed)         │
├─────────────────────────────────────────────────────────┤
│  Stage 2: Wavelet    (residual bursts isolated)         │
├─────────────────────────────────────────────────────────┤
│  Stage 3: Butterworth (low-pass output, dashed ref.)    │
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
  across nearby samples instead of removing it, and a wavelet
  transform's coefficients would be distorted by large outliers if run
  before they're cleaned up.
- **Wavelet shrinkage** sits between Hampel and Butterworth because it
  targets what neither of the other two stages can: short, wideband
  bursts (EMG artefact) that survive Hampel's fixed local window but
  aren't confined to frequencies above Butterworth's single fixed
  cutoff. Soft-thresholding (rather than hard) avoids introducing new
  discontinuities into the waveform.
- **Butterworth low-pass** was chosen for the final smoothing stage
  over a simple moving average because:
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

*This repository is a demonstration project prepared for the
**Scientific Engineer** role at **inBiome**.*
