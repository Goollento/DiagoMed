"""
denoise_demo.py
---------------
Personal proof-of-concept demonstration of biosignal denoising.

Pipeline
--------
This mirrors how real ECG/PPG/EEG pipelines are usually built: repair
corrupted samples *before* any linear filtering runs, so the linear
filters don't "ring" on sharp discontinuities.

1. Synthesise a clean physiological-like signal (a periodic oscillation,
   similar to a pulse-oximetry waveform, plus slow baseline drift).
2. Corrupt it with a realistic, layered noise model: white Gaussian noise,
   1/f ("pink") drift-like noise, 50 Hz mains interference, short EMG-like
   high-frequency bursts (muscle artefact), abrupt baseline shifts
   (electrode/motion artefact), brief signal dropouts (contact loss), and
   sensor clipping (ADC saturation).
3. Repair, then filter, in five stages:
     a) Hampel mask -> flag isolated outlier samples (median + MAD in a
        sliding window) without altering anything yet.
     b) Mask dilation -> grow each flagged region a few samples in each
        direction, since a real artefact usually corrupts a short
        neighbourhood, not just the single most extreme sample.
     c) PCHIP interpolation -> replace every flagged sample by
        interpolating from the surrounding good samples. Repairing first
        means the linear filters that follow never see a hard
        discontinuity to ring on.
     d) Notch filter (50 Hz) -> removes mains interference specifically,
        rather than relying on the low-pass roll-off alone.
     e) Butterworth zero-phase band-pass (`filtfilt`, not `lfilter`) ->
        removes the bulk of the remaining broadband noise, above *and*
        below the physiological band -- this also removes the 0.05 Hz
        baseline drift, which is treated here as slow artefact, not
        signal.
     f) Wavelet shrinkage (soft threshold, shallow decomposition) -> mops
        up whatever short, wideband residue survives the previous
        stages, using a noise-floor estimate taken from the
        pre-filtering (post-interpolation) signal.
4. Quantify quality at every stage with SNR (dB) against the known
   ground-truth clean signal.

Author: personal pet project
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import binary_dilation
from scipy.interpolate import PchipInterpolator


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


def _generate_baseline_shifts(n: int,
                              fs: int,
                              rng: np.random.Generator,
                              shift_count: int,
                              shift_len_range_s: tuple[float, float],
                              shift_amp: float) -> np.ndarray:
    """Generate abrupt, rectangular baseline-level shifts.

    Models the sudden DC offset caused by electrode/sensor movement: the
    signal jumps to a new level and holds there for a short duration. The
    edges are deliberately *not* tapered (unlike the EMG bursts) --
    sharpness here is the point: it's exactly the kind of discontinuity
    a linear filter rings on, and what the Hampel-mask + interpolation
    stage is meant to repair before any filtering happens.
    """
    noise = np.zeros(n)
    if shift_count <= 0 or n == 0:
        return noise

    min_s, max_s = shift_len_range_s
    for _ in range(shift_count):
        length = int(rng.uniform(min_s, max_s) * fs)
        length = max(1, min(length, n))
        start = int(rng.integers(low=0, high=n - length + 1))
        sign = rng.choice([-1.0, 1.0])
        noise[start:start + length] += sign * shift_amp
    return noise


def _apply_dropouts(signal: np.ndarray,
                    fs: int,
                    rng: np.random.Generator,
                    dropout_count: int,
                    dropout_len_range_s: tuple[float, float],
                    flatline_noise_amp: float) -> np.ndarray:
    """Simulate brief signal dropouts (contact loss / sensor disconnect).

    Rather than attenuating the existing waveform, each dropout window is
    replaced with a near-flat line (the value at the start of the dropout
    plus a little residual electronic noise) -- closer to what a real
    disconnected sensor produces than a scaled-down copy of the signal.
    """
    signal = signal.copy()
    n = signal.size
    if dropout_count <= 0 or n == 0:
        return signal

    min_s, max_s = dropout_len_range_s
    for _ in range(dropout_count):
        length = int(rng.uniform(min_s, max_s) * fs)
        length = max(1, min(length, n))
        start = int(rng.integers(low=0, high=n - length + 1))
        hold_value = signal[start]
        signal[start:start + length] = (
            hold_value + rng.normal(scale=flatline_noise_amp, size=length)
        )
    return signal


def add_realistic_noise(clean: np.ndarray,
                        fs: int,
                        gaussian_amp: float = 0.3,
                        pink_amp: float = 0.25,
                        mains_freq: float = 50.0,
                        mains_amp: float = 0.2,
                        emg_burst_count: int = 3,
                        emg_burst_duration_s: float = 0.3,
                        emg_burst_amp: float = 1.2,
                        baseline_shift_count: int = 3,
                        baseline_shift_len_range_s: tuple[float, float] = (0.15, 0.4),
                        baseline_shift_amp: float = 1.5,
                        dropout_count: int = 3,
                        dropout_len_range_s: tuple[float, float] = (0.1, 0.3),
                        dropout_flatline_noise_amp: float = 0.02,
                        clip_limit: float | None = 3.0,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Corrupt a clean signal with a layered, realistic noise model: white
    Gaussian noise + 1/f pink noise + 50 Hz mains interference + EMG-like
    high-frequency bursts + abrupt baseline shifts + brief dropouts +
    sensor clipping.

    Baseline shifts, dropouts and clipping replace what used to be smooth,
    Hann-windowed "motion artefact" pulses: real motion/contact artefacts
    tend to look like sharp level jumps, flatlines, or railed peaks rather
    than smooth bumps, and a repair-then-filter pipeline is specifically
    meant to be tested against discontinuities like these.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if min(gaussian_amp, pink_amp, mains_amp, emg_burst_amp,
          baseline_shift_amp) < 0:
        raise ValueError("noise amplitudes must be non-negative")
    if emg_burst_count < 0 or baseline_shift_count < 0 or dropout_count < 0:
        raise ValueError("burst/shift/dropout counts must be non-negative")
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
    baseline = _generate_baseline_shifts(n, fs, rng, baseline_shift_count,
                                         baseline_shift_len_range_s,
                                         baseline_shift_amp)

    noisy = clean + gaussian + pink + mains + emg + baseline

    noisy = _apply_dropouts(noisy, fs, rng, dropout_count,
                            dropout_len_range_s, dropout_flatline_noise_amp)

    if clip_limit is not None:
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {clip_limit}")
        noisy = np.clip(noisy, -clip_limit, clip_limit)

    return noisy


# ---------------------------------------------------------------------------
# Stage 1 of the pipeline: Hampel outlier mask (detection only, no repair)
# ---------------------------------------------------------------------------
def hampel_mask(signal: np.ndarray,
               window_size: int = 11,
               n_sigmas: float = 2.5) -> np.ndarray:
    """Flag outlier samples using a sliding-window Hampel criterion.

    For each sample, the median and the Median Absolute Deviation (MAD) of
    a surrounding window are computed. A sample is flagged if it deviates
    from the local median by more than `n_sigmas` scaled-MADs. Unlike a
    Hampel *filter*, this function does not alter the signal -- it only
    returns a boolean mask of which samples look corrupted, so a separate
    repair step (dilation + interpolation) can act on exactly those
    samples and nothing else.

    Parameters
    ----------
    signal : np.ndarray
        Input samples (typically the raw noisy signal).
    window_size : int
        Size of the sliding window in samples (should be odd; an even
        value is rounded down to the nearest valid half-window).
    n_sigmas : float
        Rejection threshold in scaled-MAD units.

    Returns
    -------
    np.ndarray
        Boolean mask, True where a sample is flagged as an outlier, same
        length as `signal`.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if n_sigmas <= 0:
        raise ValueError(f"n_sigmas must be positive, got {n_sigmas}")

    n = signal.size
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    half = max(1, window_size // 2)
    k = 1.4826  # scale factor so MAD approximates std for Gaussian data

    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        window = signal[left:right]

        median = np.median(window)
        mad = k * np.median(np.abs(window - median))
        if mad == 0:
            continue

        if abs(signal[i] - median) > n_sigmas * mad:
            mask[i] = True

    return mask


# ---------------------------------------------------------------------------
# Stage 2 of the pipeline: repair via PCHIP interpolation over flagged samples
# ---------------------------------------------------------------------------
def interpolate_mask(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace flagged (masked) samples via PCHIP interpolation.

    Every sample where `mask` is True is discarded and re-estimated from
    the surrounding "good" samples using a Piecewise Cubic Hermite
    Interpolating Polynomial (PCHIP). PCHIP is preferred over a cubic
    spline here because it does not overshoot between points -- important
    right after a large corrupted region has been removed, where a
    regular spline can ring and introduce new, artificial extrema.

    Parameters
    ----------
    signal : np.ndarray
        Input samples, some of which are corrupted.
    mask : np.ndarray
        Boolean mask, True where `signal` should be discarded and
        interpolated over (typically the dilated output of
        `hampel_mask`). Must be the same length as `signal`.

    Returns
    -------
    np.ndarray
        Signal with flagged samples replaced by interpolated values, same
        length as input.
    """
    signal = np.asarray(signal, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if mask.shape != signal.shape:
        raise ValueError(
            f"mask must match signal's shape: got {mask.shape} vs {signal.shape}"
        )

    n = signal.size
    if n == 0 or not mask.any():
        return signal.copy()

    good = ~mask
    if good.sum() < 2:
        raise ValueError(
            "fewer than 2 unflagged samples remain -- cannot interpolate; "
            "loosen the Hampel threshold or reduce dilation"
        )

    x = np.arange(n)
    interpolator = PchipInterpolator(x[good], signal[good], extrapolate=True)
    repaired = signal.copy()
    repaired[mask] = interpolator(x[mask])
    return repaired


# ---------------------------------------------------------------------------
# Stage 3 of the pipeline: notch filter (mains interference)
# ---------------------------------------------------------------------------
def notch_filter(signal: np.ndarray,
                 fs: int,
                 freq: float = 50.0,
                 q: float = 30.0) -> np.ndarray:
    """Remove a narrow-band interference tone (mains hum) with a zero-phase
    IIR notch filter.

    A notch targets the mains frequency specifically, rather than relying
    on the broader roll-off of the low-pass/band-pass stage that follows.
    Use `freq=60.0` for recordings made on a 60 Hz mains grid (e.g. North
    America).

    Parameters
    ----------
    signal : np.ndarray
        Input samples.
    fs : int
        Sampling frequency in Hz.
    freq : float
        Frequency to notch out, in Hz (50 for Europe, 60 for the US).
    q : float
        Quality factor: higher values give a narrower notch.

    Returns
    -------
    np.ndarray
        Filtered signal, same length as input.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    nyquist = 0.5 * fs
    if not 0 < freq < nyquist:
        raise ValueError(
            f"freq must satisfy 0 < freq < fs/2 (got freq={freq}, fs/2={nyquist})"
        )
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")

    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Stage 4 of the pipeline: Butterworth zero-phase band-pass
# ---------------------------------------------------------------------------
def denoise_signal(noisy_signal: np.ndarray,
                   fs: int = 100,
                   low_cutoff_hz: float | None = 0.5,
                   high_cutoff_hz: float = 3.5,
                   order: int = 4) -> np.ndarray:
    """Remove the bulk of the remaining broadband noise with a zero-phase
    Butterworth filter, band-pass by default.

    A pure low-pass only rejects energy *above* `high_cutoff_hz`. It does
    nothing about noise sitting *below* the signal band -- and slow
    baseline shifts / drift have most of their energy well under a
    low-pass cutoff, so they sail through a low-pass filter untouched.
    Turning the filter into a band-pass (`low_cutoff_hz` to
    `high_cutoff_hz`) closes that gap: with the 1.2 Hz physiological
    oscillation as the signal of interest, a passband of roughly
    0.5-3.5 Hz rejects both the high-frequency noise (mains, Gaussian/pink,
    EMG) *and* the slow drift below the signal band -- including the
    0.05 Hz baseline drift added to the synthetic "clean" signal itself,
    which is treated here as slow artefact, not informative content.

    Pass `low_cutoff_hz=None` to fall back to a pure low-pass filter, e.g.
    for signals where slow baseline wander is part of what you want to
    keep (temperature trends, slow drug-response curves) rather than
    noise to reject.

    By the time this stage runs, sharp discontinuities (spikes, baseline
    shifts, dropouts) have already been repaired by the Hampel-mask +
    interpolation stage, so `filtfilt` has nothing sharp left to ring on.

    Parameters
    ----------
    noisy_signal : np.ndarray
        Input samples to be cleaned (ideally already repaired via
        `hampel_mask` + `interpolate_mask`, and passed through
        `notch_filter`).
    fs : int
        Sampling frequency of `noisy_signal` in Hz.
    low_cutoff_hz : float or None
        Lower -3 dB cut-off of the passband. None disables the high-pass
        side and falls back to a pure low-pass filter.
    high_cutoff_hz : float
        Upper -3 dB cut-off of the passband (or of the low-pass, if
        `low_cutoff_hz` is None).
    order : int
        Order of the Butterworth filter. For a band-pass, `scipy.signal.butter`
        actually returns a filter with 2x this many poles (one low-pass and
        one high-pass prototype combined), so the effective roll-off is
        steeper than the same `order` used in low-pass mode.

    Returns
    -------
    np.ndarray
        Cleaned signal, same length as the input.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    nyquist = 0.5 * fs
    if not 0 < high_cutoff_hz < nyquist:
        raise ValueError(
            f"high_cutoff_hz must satisfy 0 < high_cutoff_hz < fs/2 "
            f"(got high_cutoff_hz={high_cutoff_hz}, fs/2={nyquist})"
        )
    if low_cutoff_hz is not None and not 0 < low_cutoff_hz < high_cutoff_hz:
        raise ValueError(
            f"low_cutoff_hz must satisfy 0 < low_cutoff_hz < high_cutoff_hz "
            f"(got low_cutoff_hz={low_cutoff_hz}, high_cutoff_hz={high_cutoff_hz})"
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

    if low_cutoff_hz is None:
        b, a = butter(order, high_cutoff_hz / nyquist, btype="low", analog=False)
    else:
        b, a = butter(order, [low_cutoff_hz / nyquist, high_cutoff_hz / nyquist],
                      btype="band", analog=False)

    # Compute padlen from the actual filter coefficients rather than a fixed
    # formula -- a band-pass filter of the same `order` has twice as many
    # coefficients as a low-pass one, so the minimum signal length differs.
    padlen = 3 * max(len(a), len(b))
    if noisy_signal.size <= padlen:
        raise ValueError(
            f"signal length ({noisy_signal.size}) must exceed "
            f"3*max(len(a),len(b)) = {padlen} samples for filtfilt; "
            f"use a longer recording or a lower order"
        )

    cleaned = filtfilt(b, a, noisy_signal)
    return cleaned


# ---------------------------------------------------------------------------
# Stage 5 of the pipeline: wavelet-domain denoising (residual bursts)
# ---------------------------------------------------------------------------
def wavelet_denoise(signal: np.ndarray,
                    noise_reference: np.ndarray | None = None,
                    wavelet: str = "sym8",
                    level: int | None = 4,
                    mode: str = "soft") -> np.ndarray:
    """Denoise via multi-resolution wavelet shrinkage (VisuShrink-style).

    Run last in the pipeline, after interpolation, notch filtering and
    Butterworth band-pass have already repaired discontinuities and
    removed the bulk of the noise. What's left at that point is whatever
    short, wideband residue survives -- exactly what a fixed-window
    repair step or a fixed frequency passband can't fully isolate on
    their own, since wavelet decomposition localizes energy in time *and*
    frequency simultaneously.

    The signal is decomposed with a discrete wavelet transform, the detail
    (high-frequency) coefficients at each level are shrunk with a universal
    (VisuShrink) threshold, and the signal is reconstructed.
    Soft-thresholding shrinks every coefficient toward zero and is used
    here rather than hard-thresholding, which zeroes coefficients outright
    and introduces small discontinuities ("kinks") into the reconstructed
    waveform -- visible as jagged edges on a physiological signal that
    should be smooth.

    Threshold estimation is decoupled from what gets filtered: by the time
    the notch and Butterworth stages have run, the finest wavelet detail
    coefficients of `signal` are mostly gone, so estimating sigma from
    `signal` itself yields a threshold too small to do anything useful.
    Passing `noise_reference` -- the signal *before* those stages (i.e.
    the output of `interpolate_mask`), which still has its high
    frequencies intact -- fixes this: sigma is estimated from
    `noise_reference`'s finest detail coefficients, but that threshold is
    applied to `signal`'s own coefficients for the actual shrinkage.

    Parameters
    ----------
    signal : np.ndarray
        Input samples to denoise (e.g. output of `denoise_signal`).
    noise_reference : np.ndarray or None
        Signal to estimate the noise threshold from -- ideally the output
        of `interpolate_mask`, taken before the notch/Butterworth stages,
        since it still carries the high-frequency content the threshold
        estimate needs. Must be the same length as `signal`. If None,
        `signal` is used for both roles (not recommended in this
        pipeline, since the threshold collapses to near zero).
    wavelet : str
        Wavelet family. Smooth, symmetric wavelets such as "sym8", "sym4"
        or "db4" are preferred here: their shape is closer to a
        physiological waveform than sharper wavelets like "db2" or "haar",
        so they distort the underlying sine less while still catching
        transient bursts.
    level : int or None
        Decomposition depth. Kept shallow (3-5) by default: decomposing
        too deep starts to fold the signal's own carrier frequency into
        the detail coefficients, so shrinkage ends up attenuating the
        physiological oscillation itself, not just the artefact bursts.
        None falls back to the maximum depth the signal length and
        wavelet support (not recommended for this reason).
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

    if noise_reference is None:
        noise_reference = signal
    else:
        noise_reference = np.asarray(noise_reference, dtype=float)
        if noise_reference.ndim != 1:
            raise ValueError(
                f"noise_reference must be 1-D, got shape {noise_reference.shape}"
            )
        if noise_reference.shape != signal.shape:
            raise ValueError(
                f"noise_reference must match signal's shape: "
                f"got {noise_reference.shape} vs signal {signal.shape}"
            )

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

    # Universal (VisuShrink) threshold: noise sigma estimated robustly, via
    # MAD, from the finest-detail coefficients of the *reference* signal
    # (pre-notch/pre-Butterworth), threshold = sigma * sqrt(2 ln n). This is
    # the only place `noise_reference` is used -- the shrinkage itself
    # still operates on `signal`'s own coefficients below.
    ref_coeffs = pywt.wavedec(noise_reference, wavelet, level=level)
    finest_ref_detail = ref_coeffs[-1]
    sigma = np.median(np.abs(finest_ref_detail)) / 0.6745 if finest_ref_detail.size else 0.0
    uthresh = sigma * np.sqrt(2.0 * np.log(n)) if sigma > 0 else 0.0

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=mode) for c in detail_coeffs
    ]
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:n]  # waverec can pad by 1 sample depending on length parity


# ---------------------------------------------------------------------------
# Quality metric (SNR)
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
                 after_interp: np.ndarray,
                 after_notch: np.ndarray,
                 after_butterworth: np.ndarray,
                 final_cleaned: np.ndarray,
                 snr_noisy: float,
                 snr_interp: float,
                 snr_notch: float,
                 snr_butterworth: float,
                 snr_final: float,
                 save_path: str | None = "denoise_demo.png") -> None:
    """Plot clean / noisy / repaired / notch / Butterworth / final traces,
    annotated with the SNR (dB) achieved at each stage, and optionally save
    to disk.
    """
    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(t, clean, color="#1b7f3a", linewidth=1.4, label="Clean signal")
    axes[0].set_title("Clean synthetic biosignal (sine + baseline drift)")
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, noisy, color="#b03030", linewidth=0.8,
                 label="Noisy (Gaussian + pink + 50 Hz + EMG + shifts + dropouts + clipping)")
    axes[1].set_title(f"Noisy signal  |  SNR = {snr_noisy:.2f} dB")
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, after_interp, color="#c47f00", linewidth=0.9,
                 label="After Hampel mask + dilation + PCHIP interpolation")
    axes[2].set_title(f"Stage 1: Repair (mask + interpolate)  |  SNR = {snr_interp:.2f} dB")
    axes[2].set_ylabel("Amplitude (a.u.)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, after_notch, color="#a1621f", linewidth=0.9,
                 label="After 50 Hz notch filter")
    axes[3].set_title(f"Stage 2: Notch filter (50 Hz)  |  SNR = {snr_notch:.2f} dB")
    axes[3].set_ylabel("Amplitude (a.u.)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(alpha=0.3)

    axes[4].plot(t, after_butterworth, color="#1f4ea1", linewidth=0.9,
                 label="After Butterworth band-pass (0.5-3.5 Hz, zero-phase)")
    axes[4].set_title(f"Stage 3: Butterworth band-pass  |  SNR = {snr_butterworth:.2f} dB")
    axes[4].set_ylabel("Amplitude (a.u.)")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].grid(alpha=0.3)

    axes[5].plot(t, final_cleaned, color="#7a3fa0", linewidth=1.4,
                 label="Final: repair + notch + Butterworth + wavelet")
    axes[5].plot(t, clean, color="#1b7f3a", linewidth=1.0,
                 linestyle="--", alpha=0.6, label="Reference (clean)")
    axes[5].set_title(f"Stage 4: Wavelet shrinkage  |  SNR = {snr_final:.2f} dB")
    axes[5].set_xlabel("Time (s)")
    axes[5].set_ylabel("Amplitude (a.u.)")
    axes[5].legend(loc="upper right", fontsize=8)
    axes[5].grid(alpha=0.3)

    fig.suptitle("Biosignal denoising pipeline: repair -> notch -> Butterworth -> wavelet -> SNR",
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
    fs = 250              # sampling frequency in Hz (headroom above the 50 Hz notch)
    duration_s = 10.0      # total length of the recording

    t, clean = generate_clean_signal(duration_s=duration_s, fs=fs)
    noisy = add_realistic_noise(clean, fs=fs)

    # --- Denoising pipeline: repair -> notch -> Butterworth -> wavelet ---
    # 1) Find outlier samples (spikes, baseline-shift edges) without
    #    touching the signal yet.
    mask = hampel_mask(noisy, window_size=11, n_sigmas=2.5)
    # 2) Grow each flagged region: a real artefact usually corrupts a
    #    short neighbourhood, not just the single most extreme sample.
    mask = binary_dilation(mask, iterations=3)
    # 3) Repair via PCHIP interpolation -- discontinuities are gone before
    #    any linear filter runs, so nothing is left to "ring" on.
    after_interp = interpolate_mask(noisy, mask)
    # 4) Remove 50 Hz mains hum specifically.
    after_notch = notch_filter(after_interp, fs=fs, freq=50.0, q=30.0)
    # 5) Band-pass (0.5-3.5 Hz): removes broadband noise on both sides of
    #    the 1.2 Hz physiological band, including slow baseline drift.
    #    filtfilt is zero-phase (forward+backward), so this introduces no
    #    time shift -- safe here since this is offline post-processing,
    #    not a real-time/streaming pipeline.
    after_butterworth = denoise_signal(after_notch, fs=fs,
                                       low_cutoff_hz=0.5, high_cutoff_hz=3.5,
                                       order=4)
    # 6) Wavelet shrinkage mops up residual bursts; its noise threshold is
    #    estimated from `after_interp` (before notch/Butterworth), since
    #    that's the last point where the signal still has its full
    #    high-frequency noise character intact.
    final_cleaned = wavelet_denoise(after_butterworth, noise_reference=after_interp,
                                    wavelet="sym8", level=4, mode="soft")

    snr_noisy = compute_snr_db(clean, noisy)
    snr_interp = compute_snr_db(clean, after_interp)
    snr_notch = compute_snr_db(clean, after_notch)
    snr_butterworth = compute_snr_db(clean, after_butterworth)
    snr_final = compute_snr_db(clean, final_cleaned)

    print(f"Samples flagged by Hampel mask (pre-dilation... shown post-dilation): {mask.sum()} / {mask.size}")
    print(f"SNR of noisy signal:                                {snr_noisy:6.2f} dB")
    print(f"SNR after repair (mask + interpolate):               {snr_interp:6.2f} dB")
    print(f"SNR after repair + notch:                            {snr_notch:6.2f} dB")
    print(f"SNR after repair + notch + Butterworth:               {snr_butterworth:6.2f} dB")
    print(f"SNR after full pipeline (+ wavelet):                  {snr_final:6.2f} dB")
    print(f"Total improvement:                                   {snr_final - snr_noisy:+.2f} dB")

    plot_results(t, clean, noisy, after_interp, after_notch, after_butterworth,
                final_cleaned, snr_noisy, snr_interp, snr_notch,
                snr_butterworth, snr_final)


if __name__ == "__main__":
    main()
