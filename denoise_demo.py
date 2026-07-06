"""
denoise_demo.py
---------------
Personal proof-of-concept demonstration of biosignal denoising.

Pipeline
--------
1. Synthesise a clean physiological-like signal (a periodic oscillation,
   similar to a pulse-oximetry waveform, plus slow baseline drift).
2. Corrupt it with a realistic, layered noise model: white Gaussian noise,
   1/f ("pink") drift-like noise, 50 Hz mains interference, short EMG-like
   high-frequency bursts (muscle artefact) and motion-artefact pulses
   (electrode pops / contact loss).
3. Denoise in three stages:
     a) Hampel filter -> removes isolated outliers / spikes (motion
        artefacts, electrode pops) without touching the rest of the
        waveform (a nonlinear, robust step).
     b) Wavelet shrinkage (VisuShrink, soft threshold) -> isolates
        short, wideband bursts (residual motion/EMG artefact) in the
        time-frequency domain, which a fixed-window Hampel filter and a
        fixed-cutoff Butterworth filter both miss on their own.
     c) Butterworth zero-phase low-pass (`filtfilt`, not `lfilter`) ->
        removes the remaining broadband/high-frequency noise with no
        phase distortion -- safe here since this is offline
        post-processing, not a real-time/streaming pipeline.
4. Quantify quality at every stage with SNR (dB) against the known
   ground-truth clean signal.

Author: personal pet project
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pywt
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
        Frequency of the main oscillation in Hz (~1.2 Hz ~= 72 bpm).

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
            f"({0.5 * fs} Hz) -- aliasing would corrupt the signal"
        )

    t = np.linspace(0.0, duration_s, int(duration_s * fs), endpoint=False)
    oscillation = np.sin(2 * np.pi * signal_freq * t)
    baseline_drift = 0.5 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz drift
    clean = oscillation + baseline_drift
    return t, clean


# ---------------------------------------------------------------------------
# Noise components (composable, so the overall noise model is layered)
# ---------------------------------------------------------------------------
def _generate_pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate unit-variance 1/f ("pink") noise via spectral shaping.

    Pink noise is a reasonable stand-in for slow, correlated physiological
    and electrode-drift noise that white Gaussian noise does not capture.
    """
    if n <= 1:
        return np.zeros(n)
    white = rng.normal(size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid division by zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    std = np.std(pink)
    if std > 0:
        pink = pink / std
    return pink


def _generate_emg_bursts(n: int,
                         fs: int,
                         rng: np.random.Generator,
                         burst_count: int,
                         burst_duration_s: float,
                         burst_amp: float) -> np.ndarray:
    """Generate short bursts of high-frequency noise mimicking muscle (EMG)
    artefact, e.g. from patient movement or shivering.
    """
    noise = np.zeros(n)
    burst_len = int(burst_duration_s * fs)
    if burst_count <= 0 or burst_len < 1 or n <= burst_len:
        return noise

    starts = rng.integers(low=0, high=n - burst_len, size=burst_count)
    envelope = np.hanning(burst_len)
    for start in starts:
        burst = rng.normal(scale=burst_amp, size=burst_len) * envelope
        noise[start:start + burst_len] += burst
    return noise


def _generate_motion_artifacts(n: int,
                               rng: np.random.Generator,
                               artifact_count: int,
                               artifact_len_range: tuple[int, int],
                               artifact_amp: float) -> np.ndarray:
    """Generate rectangular-ish pulses mimicking motion artefacts / electrode
    pops: short segments where the signal jumps to an offset value.
    """
    noise = np.zeros(n)
    if artifact_count <= 0 or n == 0:
        return noise

    min_len, max_len = artifact_len_range
    for _ in range(artifact_count):
        length = int(rng.integers(low=min_len, high=max_len + 1))
        if length < 1 or length >= n:
            continue
        start = int(rng.integers(low=0, high=n - length))
        sign = rng.choice([-1.0, 1.0])
        # Smooth-ish rectangular pulse (half-sine ramp in/out) so the
        # Hampel filter sees it as a genuine run of outliers, not one spike.
        ramp = np.hanning(length)
        noise[start:start + length] += sign * artifact_amp * ramp
    return noise


def add_realistic_noise(clean: np.ndarray,
                        fs: int,
                        gaussian_amp: float = 0.3,
                        pink_amp: float = 0.25,
                        mains_freq: float = 50.0,
                        mains_amp: float = 0.2,
                        emg_burst_count: int = 3,
                        emg_burst_duration_s: float = 0.3,
                        emg_burst_amp: float = 1.2,
                        motion_artifact_count: int = 5,
                        motion_artifact_len_range: tuple[int, int] = (3, 10),
                        motion_artifact_amp: float = 2.2,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Corrupt a clean signal with a layered, more realistic noise model:
    white Gaussian noise + 1/f pink noise + 50 Hz mains interference +
    EMG-like high-frequency bursts + motion-artefact pulses.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if min(gaussian_amp, pink_amp, mains_amp, emg_burst_amp, motion_artifact_amp) < 0:
        raise ValueError("noise amplitudes must be non-negative")
    if emg_burst_count < 0 or motion_artifact_count < 0:
        raise ValueError("burst/artifact counts must be non-negative")
    if mains_freq < 0:
        raise ValueError(f"mains_freq must be non-negative, got {mains_freq}")
    if mains_freq > 0.5 * fs and mains_amp > 0:
        # Above Nyquist the mains sinusoid would alias to a different
        # frequency -- refuse rather than silently produce a misleading artefact.
        raise ValueError(
            f"mains_freq ({mains_freq} Hz) is above Nyquist "
            f"({0.5 * fs} Hz); raise fs or set mains_amp=0"
        )

    if rng is None:
        rng = np.random.default_rng(seed=42)

    n = clean.size
    t = np.arange(n) / fs

    gaussian = rng.normal(loc=0.0, scale=gaussian_amp, size=n)
    pink = pink_amp * _generate_pink_noise(n, rng)
    mains = mains_amp * np.sin(2 * np.pi * mains_freq * t)
    emg = _generate_emg_bursts(n, fs, rng, emg_burst_count,
                               emg_burst_duration_s, emg_burst_amp)
    motion = _generate_motion_artifacts(n, rng, motion_artifact_count,
                                        motion_artifact_len_range,
                                        motion_artifact_amp)

    return clean + gaussian + pink + mains + emg + motion


# ---------------------------------------------------------------------------
# Stage 1 of denoising: Hampel filter (robust outlier / spike removal)
# ---------------------------------------------------------------------------
def hampel_filter(signal: np.ndarray,
                  window_size: int = 7,
                  n_sigmas: float = 3.0) -> np.ndarray:
    """Remove outliers/spikes with a sliding-window Hampel filter.

    For each sample, the median and the Median Absolute Deviation (MAD) of a
    surrounding window are computed. If a sample deviates from the local
    median by more than `n_sigmas` scaled-MADs, it is replaced by the local
    median. This is a robust, nonlinear step that targets isolated spikes
    and short artefact runs (e.g. motion artefacts) without smoothing or
    phase-shifting the rest of the waveform the way a linear filter would.

    Parameters
    ----------
    signal : np.ndarray
        Input samples (typically the raw noisy signal).
    window_size : int
        Size of the sliding window in samples (should be odd; an even
        value is rounded up internally).
    n_sigmas : float
        Rejection threshold in scaled-MAD units.

    Returns
    -------
    np.ndarray
        Signal with outliers replaced by local medians, same length as input.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if n_sigmas <= 0:
        raise ValueError(f"n_sigmas must be positive, got {n_sigmas}")

    n = signal.size
    if n == 0:
        return signal.copy()

    half_window = max(1, window_size // 2)
    k = 1.4826  # scale factor so MAD approximates std for Gaussian data
    cleaned = signal.copy()

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = signal[start:end]
        median = np.median(window)
        mad = k * np.median(np.abs(window - median))
        if mad == 0:
            continue
        if np.abs(signal[i] - median) > n_sigmas * mad:
            cleaned[i] = median

    return cleaned


# ---------------------------------------------------------------------------
# Stage 2 of denoising: wavelet-domain denoising (motion-artefact / EMG bursts)
# ---------------------------------------------------------------------------
def wavelet_denoise(signal: np.ndarray,
                    wavelet: str = "db6",
                    level: int | None = None,
                    mode: str = "soft") -> np.ndarray:
    """Denoise via multi-resolution wavelet shrinkage (VisuShrink-style).

    Hampel and Butterworth are both "global" in a sense: Hampel only sees a
    fixed-size local window, and Butterworth applies the same frequency
    cutoff everywhere in time. Neither of them isolates a burst of motion
    artefact / EMG noise that spans a wide range of frequencies over a
    short time window -- that is exactly what wavelet decomposition is
    good at, since it separates the signal into localized time-frequency
    components.

    The signal is decomposed with a discrete wavelet transform, the detail
    (high-frequency) coefficients at each level are shrunk with a universal
    (VisuShrink) threshold estimated from the finest-level noise, and the
    signal is reconstructed. Soft-thresholding shrinks every coefficient
    toward zero (denoising the whole record); it does not "cut out" a time
    segment the way notching a channel would -- for that you'd want a
    reference-based method (adaptive filtering / ICA), which needs an
    extra noise-reference channel this single-channel demo does not have.

    Parameters
    ----------
    signal : np.ndarray
        Input samples (e.g. output of `hampel_filter`).
    wavelet : str
        Wavelet family (e.g. "db6", "sym8", "coif4").
    level : int or None
        Decomposition depth. None picks the maximum sensible depth for the
        given signal length and wavelet.
    mode : str
        Thresholding mode passed to `pywt.threshold` ("soft" or "hard").
        Soft is preferred: it avoids the discontinuities hard-thresholding
        introduces, which matter for a physiological waveform.

    Returns
    -------
    np.ndarray
        Denoised signal, same length as input.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if mode not in ("soft", "hard"):
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")

    n = signal.size
    if n == 0:
        return signal.copy()

    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    if max_level < 1:
        raise ValueError(
            f"signal length ({n}) is too short for wavelet '{wavelet}'"
        )
    if level is None:
        level = max_level
    elif not 1 <= level <= max_level:
        raise ValueError(
            f"level must be in [1, {max_level}] for this signal/wavelet, got {level}"
        )

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    detail_coeffs = coeffs[1:]

    # Universal (VisuShrink) threshold: noise sigma estimated robustly from
    # the finest-detail coefficients via MAD, threshold = sigma * sqrt(2 ln n).
    finest_detail = detail_coeffs[-1]
    sigma = np.median(np.abs(finest_detail)) / 0.6745 if finest_detail.size else 0.0
    uthresh = sigma * np.sqrt(2.0 * np.log(n)) if sigma > 0 else 0.0

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=mode) for c in detail_coeffs
    ]
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:n]  # waverec can pad by 1 sample depending on length parity


# ---------------------------------------------------------------------------
# Stage 3 of denoising: Butterworth zero-phase low-pass
# ---------------------------------------------------------------------------
def denoise_signal(noisy_signal: np.ndarray,
                   fs: int = 100,
                   cutoff_hz: float = 4.0,
                   order: int = 4) -> np.ndarray:
    """Remove remaining high-frequency/broadband noise with a zero-phase
    Butterworth low-pass.

    A low-pass filter is well suited to physiological signals whose
    informative content lies below a few Hz (heart rate, respiration,
    slow temperature changes), while the dominant remaining noise sources
    (mains hum, EMG bursts, broadband Gaussian/pink noise) sit above the
    chosen cutoff and are attenuated.

    `filtfilt` applies the filter forward and backward to avoid the
    phase distortion that would shift the waveform in time -- important
    for diagnostic interpretation.

    Parameters
    ----------
    noisy_signal : np.ndarray
        Input samples to be cleaned (ideally already passed through
        `hampel_filter` to remove spikes first).
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
        raise ValueError("noisy_signal contains NaN or Inf -- clean the input first")

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
# Stage 4: quality metric (SNR)
# ---------------------------------------------------------------------------
def compute_snr_db(reference: np.ndarray, test_signal: np.ndarray) -> float:
    """Compute the Signal-to-Noise Ratio, in dB, of `test_signal` against a
    known ground-truth `reference`.

    SNR = 10 * log10( power(reference) / power(reference - test_signal) )

    Higher is better; this is the standard way to quantify how close a
    denoised (or noisy) signal is to the ground truth when the ground
    truth is known, as it is here since the clean signal was synthesised.
    """
    reference = np.asarray(reference, dtype=float)
    test_signal = np.asarray(test_signal, dtype=float)
    if reference.shape != test_signal.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} vs "
            f"test_signal {test_signal.shape}"
        )

    noise = reference - test_signal
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_results(t: np.ndarray,
                 clean: np.ndarray,
                 noisy: np.ndarray,
                 hampel_cleaned: np.ndarray,
                 wavelet_cleaned: np.ndarray,
                 final_cleaned: np.ndarray,
                 snr_noisy: float,
                 snr_hampel: float,
                 snr_wavelet: float,
                 snr_final: float,
                 save_path: str | None = "denoise_demo.png") -> None:
    """Plot clean / noisy / Hampel-stage / wavelet-stage / final-stage traces,
    annotated with the SNR (dB) achieved at each stage, and optionally save
    to disk.
    """
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(t, clean, color="#1b7f3a", linewidth=1.4, label="Clean signal")
    axes[0].set_title("Clean synthetic biosignal (sine + baseline drift)")
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, noisy, color="#b03030", linewidth=0.8,
                 label="Noisy (Gaussian + pink + 50 Hz + EMG bursts + motion)")
    axes[1].set_title(f"Noisy signal  |  SNR = {snr_noisy:.2f} dB")
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, hampel_cleaned, color="#c47f00", linewidth=0.9,
                 label="After Hampel filter (spikes/artefacts removed)")
    axes[2].set_title(f"Stage 1: Hampel filter  |  SNR = {snr_hampel:.2f} dB")
    axes[2].set_ylabel("Amplitude (a.u.)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, wavelet_cleaned, color="#7a3fa0", linewidth=0.9,
                 label="After wavelet shrinkage (motion/EMG isolated)")
    axes[3].set_title(f"Stage 2: Wavelet denoising  |  SNR = {snr_wavelet:.2f} dB")
    axes[3].set_ylabel("Amplitude (a.u.)")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    axes[4].plot(t, final_cleaned, color="#1f4ea1", linewidth=1.4,
                 label="After Hampel + wavelet + Butterworth low-pass")
    axes[4].plot(t, clean, color="#1b7f3a", linewidth=1.0,
                 linestyle="--", alpha=0.6, label="Reference (clean)")
    axes[4].set_title(f"Stage 3: Butterworth low-pass (zero-phase)  |  SNR = {snr_final:.2f} dB")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Amplitude (a.u.)")
    axes[4].legend(loc="upper right")
    axes[4].grid(alpha=0.3)

    fig.suptitle("Biosignal denoising pipeline: Hampel -> Wavelet -> Butterworth -> SNR",
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

    # --- Denoising pipeline: Hampel -> wavelet -> Butterworth -> SNR ---
    # Hampel tuned tighter (smaller window, lower sigma threshold) so it
    # cuts spikes/motion artefacts harder before anything else runs.
    after_hampel = hampel_filter(noisy, window_size=5, n_sigmas=2.5)
    after_wavelet = wavelet_denoise(after_hampel, wavelet="db6", mode="soft")
    # filtfilt is zero-phase (forward+backward), so this stage introduces
    # no time shift -- safe to use here since this is offline post-processing,
    # not a real-time/streaming pipeline.
    final_cleaned = denoise_signal(after_wavelet, fs=fs, cutoff_hz=4.0, order=4)

    snr_noisy = compute_snr_db(clean, noisy)
    snr_hampel = compute_snr_db(clean, after_hampel)
    snr_wavelet = compute_snr_db(clean, after_wavelet)
    snr_final = compute_snr_db(clean, final_cleaned)

    print(f"SNR of noisy signal:                         {snr_noisy:6.2f} dB")
    print(f"SNR after Hampel filter:                      {snr_hampel:6.2f} dB")
    print(f"SNR after Hampel + wavelet:                    {snr_wavelet:6.2f} dB")
    print(f"SNR after Hampel + wavelet + Butterworth:       {snr_final:6.2f} dB")
    print(f"Total improvement:                            {snr_final - snr_noisy:+.2f} dB")

    plot_results(t, clean, noisy, after_hampel, after_wavelet, final_cleaned,
                snr_noisy, snr_hampel, snr_wavelet, snr_final)


if __name__ == "__main__":
    main()
