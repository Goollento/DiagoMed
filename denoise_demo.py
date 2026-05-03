"""
denoise_demo.py
---------------
Proof-of-concept demonstration of biosignal denoising for inBiome.

The script synthesises a clean physiological-like signal (e.g. a slow
oscillation similar to a pulse-oximetry waveform plus a low-frequency
baseline drift), corrupts it with realistic noise (white Gaussian noise,
50 Hz mains interference and random spikes), and then recovers the
underlying signal using a zero-phase low-pass Butterworth filter from
SciPy. The three traces are plotted together for visual comparison.

Author: portfolio / inBiome demonstration
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def generate_clean_signal(duration_s: float = 10.0,
                          fs: int = 100,
                          signal_freq: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean synthetic biosignal.

    The signal is a sine wave (typical of a periodic physiological rhythm)
    superimposed on a slow baseline drift, which is common in real
    recordings (e.g. respiration-induced drift in PPG).

    Parameters
    ----------
    duration_s : float
        Length of the signal in seconds.
    fs : int
        Sampling frequency in Hz.
    signal_freq : float
        Frequency of the main oscillation in Hz (~1.2 Hz ≈ 72 bpm).

    Returns
    -------
    t : np.ndarray
        Time vector in seconds.
    clean : np.ndarray
        Clean signal samples.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    if signal_freq < 0:
        raise ValueError(f"signal_freq must be non-negative, got {signal_freq}")
    if signal_freq > 0.5 * fs:
        raise ValueError(
            f"signal_freq ({signal_freq} Hz) is above Nyquist "
            f"({0.5 * fs} Hz) — aliasing would corrupt the signal"
        )

    t = np.linspace(0.0, duration_s, int(duration_s * fs), endpoint=False)
    oscillation = np.sin(2 * np.pi * signal_freq * t)
    baseline_drift = 0.5 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz drift
    clean = oscillation + baseline_drift
    return t, clean


def add_realistic_noise(clean: np.ndarray,
                        fs: int,
                        gaussian_amp: float = 0.4,
                        mains_freq: float = 50.0,
                        mains_amp: float = 0.25,
                        spike_count: int = 8,
                        spike_amp: float = 2.5,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white Gaussian noise, mains interference and random spikes."""
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if gaussian_amp < 0 or mains_amp < 0 or spike_amp < 0:
        raise ValueError("noise amplitudes must be non-negative")
    if spike_count < 0:
        raise ValueError(f"spike_count must be non-negative, got {spike_count}")
    if mains_freq < 0:
        raise ValueError(f"mains_freq must be non-negative, got {mains_freq}")
    if mains_freq > 0.5 * fs and mains_amp > 0:
        # Above Nyquist the mains sinusoid would alias to a different
        # frequency — refuse rather than silently produce a misleading artefact.
        raise ValueError(
            f"mains_freq ({mains_freq} Hz) is above Nyquist "
            f"({0.5 * fs} Hz); raise fs or set mains_amp=0"
        )

    if rng is None:
        rng = np.random.default_rng(seed=42)

    n = clean.size
    t = np.arange(n) / fs

    gaussian = rng.normal(loc=0.0, scale=gaussian_amp, size=n)
    mains = mains_amp * np.sin(2 * np.pi * mains_freq * t)

    spikes = np.zeros(n)
    if spike_count > 0 and n > 0:
        spike_idx = rng.integers(low=0, high=n, size=spike_count)
        spike_signs = rng.choice([-1.0, 1.0], size=spike_count)
        spikes[spike_idx] = spike_signs * spike_amp

    return clean + gaussian + mains + spikes


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------
def denoise_signal(noisy_signal: np.ndarray,
                   fs: int = 100,
                   cutoff_hz: float = 4.0,
                   order: int = 4) -> np.ndarray:
    """Remove high-frequency noise with a zero-phase Butterworth low-pass.

    A low-pass filter is well suited to physiological signals whose
    informative content lies below a few Hz (heart rate, respiration,
    slow temperature changes), while the dominant noise sources
    (mains hum at 50/60 Hz, spikes, broadband Gaussian noise) sit
    above the chosen cutoff and are attenuated.

    `filtfilt` applies the filter forward and backward to avoid the
    phase distortion that would shift the waveform in time — important
    for diagnostic interpretation.

    Parameters
    ----------
    noisy_signal : np.ndarray
        Input samples to be cleaned.
    fs : int
        Sampling frequency of `noisy_signal` in Hz.
    cutoff_hz : float
        -3 dB cut-off frequency of the low-pass filter.
    order : int
        Order of the Butterworth filter.

    Returns
    -------
    np.ndarray
        Cleaned signal, same length as the input.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    nyquist = 0.5 * fs
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(
            f"cutoff_hz must satisfy 0 < cutoff_hz < fs/2 "
            f"(got cutoff_hz={cutoff_hz}, fs/2={nyquist})"
        )
    if not 1 <= order <= 10:
        # Higher orders quickly become numerically unstable in transfer-function
        # form; if you need a steeper roll-off, switch to SOS (sosfiltfilt).
        raise ValueError(f"order must be an integer in [1, 10], got {order}")

    noisy_signal = np.asarray(noisy_signal)
    if noisy_signal.ndim != 1:
        raise ValueError(
            f"noisy_signal must be 1-D, got shape {noisy_signal.shape}"
        )
    if not np.all(np.isfinite(noisy_signal)):
        # filtfilt propagates a single NaN/Inf to every output sample.
        raise ValueError("noisy_signal contains NaN or Inf — clean the input first")

    padlen = 3 * (order + 1)
    if noisy_signal.size <= padlen:
        raise ValueError(
            f"signal length ({noisy_signal.size}) must exceed "
            f"3*(order+1) = {padlen} samples for filtfilt; "
            f"use a longer recording or a lower order"
        )

    normalised_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalised_cutoff, btype="low", analog=False)
    cleaned = filtfilt(b, a, noisy_signal)
    return cleaned


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_results(t: np.ndarray,
                 clean: np.ndarray,
                 noisy: np.ndarray,
                 cleaned: np.ndarray,
                 save_path: str | None = "denoise_demo.png") -> None:
    """Plot the three traces stacked vertically and optionally save to disk."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, clean, color="#1b7f3a", linewidth=1.4, label="Clean signal")
    axes[0].set_title("Clean synthetic biosignal (sine + baseline drift)")
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, noisy, color="#b03030", linewidth=0.8,
                 label="Noisy signal (Gaussian + 50 Hz + spikes)")
    axes[1].set_title("Noisy signal")
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, cleaned, color="#1f4ea1", linewidth=1.4,
                 label="Cleaned signal (Butterworth low-pass, 4 Hz)")
    axes[2].plot(t, clean, color="#1b7f3a", linewidth=1.0,
                 linestyle="--", alpha=0.6, label="Reference (clean)")
    axes[2].set_title("Recovered signal after denoising")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude (a.u.)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Biosignal denoising demo — inBiome PoC",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    fs = 100               # sampling frequency in Hz
    duration_s = 10.0      # total length of the recording

    t, clean = generate_clean_signal(duration_s=duration_s, fs=fs)
    noisy = add_realistic_noise(clean, fs=fs)
    cleaned = denoise_signal(noisy, fs=fs, cutoff_hz=4.0, order=4)

    # Quality metric: residual RMS error vs. the ground-truth signal.
    rms_before = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    rms_after = float(np.sqrt(np.mean((cleaned - clean) ** 2)))
    print(f"RMS error before denoising: {rms_before:.3f}")
    print(f"RMS error after  denoising: {rms_after:.3f}")
    print(f"Improvement factor:         {rms_before / rms_after:.2f}x")

    plot_results(t, clean, noisy, cleaned)


if __name__ == "__main__":
    main()
