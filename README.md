# DiagoMed — Biosignal Denoising Demo

A small, self-contained proof-of-concept that demonstrates how a noisy
biological signal (such as a pulse-oximetry waveform or a slow temperature
trace) can be recovered using classical digital filtering.

This repository was created as a portfolio piece for **inBiome**.

---

## What the script does

`denoise_demo.py`:

1. **Generates a clean synthetic signal** — a 1.2 Hz sine wave
   (~72 bpm) superimposed on a slow 0.05 Hz baseline drift, sampled at
   100 Hz for 10 seconds.
2. **Corrupts the signal with realistic noise:**
   - White Gaussian noise (σ ≈ 0.4)
   - 50 Hz mains interference
   - A handful of random transient spikes
3. **Denoises** the signal with a 4th-order **Butterworth low-pass
   filter** (cut-off 4 Hz) applied via `scipy.signal.filtfilt`
   for zero phase distortion.
4. **Plots** the clean, noisy and recovered traces on a single figure
   and saves the result to `denoise_demo.png`.
5. Prints the RMS error before and after denoising, plus the
   improvement factor, as a simple quantitative quality check.

The denoising logic is encapsulated in:

```python
denoise_signal(noisy_signal, fs=100, cutoff_hz=4.0, order=4) -> np.ndarray
```

so it can be reused on real recordings by passing them in directly.

---

## Why this matters for medical diagnostics

Real-world physiological signals — PPG, ECG, skin temperature, EEG —
are routinely contaminated by:

- Broadband electronic noise from the acquisition hardware,
- Mains interference at 50 Hz (Europe) or 60 Hz (US),
- Motion-artefact spikes from patient movement,
- Slow baseline wander from breathing, perspiration, electrode drift.

Reliably recovering the underlying physiological waveform is the
prerequisite for any downstream diagnostic algorithm
(heart-rate detection, SpO₂ estimation, arrhythmia screening, etc.).
This demo is intentionally minimal — one classical filter, one signal
— but the structure mirrors what a production pipeline looks like.

---

## Requirements

- Python **3.9+**
- `numpy`
- `matplotlib`
- `scipy`

Install everything in one command:

```bash
pip install numpy matplotlib scipy
```

---

## How to run

From the project root:

```bash
python denoise_demo.py
```

The script will:

- Print the RMS-error metrics to the console,
- Open an interactive Matplotlib window with the three traces,
- Save the figure to `denoise_demo.png` in the current directory.

Expected console output (values are deterministic — fixed RNG seed):

```
RMS error before denoising: 0.459
RMS error after  denoising: 0.130
Improvement factor:         3.52x
```

---

## Result at a glance

The generated figure shows three stacked panels:

```
┌─────────────────────────────────────────────────────────┐
│  Clean signal       (smooth sine + slow drift)          │
├─────────────────────────────────────────────────────────┤
│  Noisy signal       (50 Hz hum + Gaussian + spikes)     │
├─────────────────────────────────────────────────────────┤
│  Cleaned signal     (low-pass output, dashed reference) │
└─────────────────────────────────────────────────────────┘
```

The bottom panel overlays the recovered trace on top of the original
ground truth (dashed) — they should be visually almost
indistinguishable.

---

## File layout

```
.
├── denoise_demo.py     # main script (signal gen + denoising + plotting)
├── README.md           # this file
└── denoise_demo.png    # produced on first run
```

---

## Notes on the choice of filter

A Butterworth low-pass was chosen over a simple moving average because:

- It has a **flat passband** — it does not attenuate the in-band
  physiological content.
- Combined with `filtfilt`, it introduces **zero phase shift**, which
  preserves the temporal alignment of features (peaks, onsets) — this
  matters for any time-based diagnostic measurement.
- The cut-off (4 Hz) sits comfortably above the dominant signal
  frequency (1.2 Hz) and well below the mains and spike content.

For more aggressive artefact rejection, a notch filter at 50/60 Hz or
a wavelet-based denoiser would be the natural next step.

---

## License

MIT — feel free to reuse for educational or evaluation purposes.

---

*This repository is a demonstration project prepared for the
**Scientific Engineer** role at **inBiome**.*
