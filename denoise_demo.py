"""
denoise_demo.py
---------------
Personal proof-of-concept demonstration of biosignal denoising.

Improved version incorporating:
- Separate true signal and baseline, SNR computed against true signal.
- NoiseGenerator class for composable noise.
- Difficulty levels (Easy, Medium, Hard, Extreme).
- Adaptive Hampel window (based on fs) and dilation.
- Adaptive interpolation (PCHIP / CubicSpline / linear based on gap length).
- Butterworth SOS filter with sosfiltfilt.
- BayesShrink wavelet denoising (instead of VisuShrink).
- Comprehensive metrics (SNR, RMSE, MAE, Pearson, PSNR, NRMSE).
- Stage contribution (delta SNR).
- FFT and PSD plots per stage.
- Configuration via dataclass.
- Pipeline class with run / plot / metrics / save.
- Reproducible experiments with seed.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.ndimage import binary_dilation
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy import signal
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
import warnings


# ---------------------------------------------------------------------------
# Enums and configuration
# ---------------------------------------------------------------------------
class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


@dataclass
class PipelineConfig:
    """Configuration for the denoising pipeline."""
    # Signal parameters
    duration_s: float = 10.0
    fs: int = 250
    signal_freq: float = 1.2          # Hz, main oscillation
    baseline_freq: float = 0.05       # Hz, slow drift

    # Noise parameters (defaults for EXTREME, will be adjusted by difficulty)
    gaussian_amp: float = 0.3
    pink_amp: float = 0.25
    mains_freq: float = 50.0
    mains_amp: float = 0.2
    emg_burst_count: int = 3
    emg_burst_duration_s: float = 0.3
    emg_burst_amp: float = 1.2
    baseline_shift_count: int = 3
    baseline_shift_len_range_s: Tuple[float, float] = (0.15, 0.4)
    baseline_shift_amp: float = 1.5
    dropout_count: int = 3
    dropout_len_range_s: Tuple[float, float] = (0.1, 0.3)
    dropout_flatline_noise_amp: float = 0.02
    clip_limit: Optional[float] = 3.0

    # Pipeline parameters
    hampel_window_sec: float = 0.1    # 100 ms
    hampel_n_sigmas: float = 2.0
    dilation_sec: float = 0.02        # 20 ms
    notch_freq: float = 50.0
    notch_q: float = 30.0
    bandpass_low: Optional[float] = 0.5   # Hz, None for low-pass only
    bandpass_high: float = 3.5
    butter_order: int = 4
    wavelet_name: str = "sym8"
    wavelet_level: int = 4
    wavelet_mode: str = "soft"

    # Experiment
    seed: int = 42
    difficulty: Difficulty = Difficulty.EXTREME
    n_repeats: int = 1               # number of noise realisations to average

    def __post_init__(self):
        # Adjust noise amplitudes based on difficulty
        if self.difficulty == Difficulty.EASY:
            self.gaussian_amp = 0.1
            self.pink_amp = 0.05
            self.mains_amp = 0.05
            self.emg_burst_amp = 0.0
            self.emg_burst_count = 0
            self.baseline_shift_amp = 0.0
            self.baseline_shift_count = 0
            self.dropout_count = 0
            self.clip_limit = None
        elif self.difficulty == Difficulty.MEDIUM:
            self.gaussian_amp = 0.15
            self.pink_amp = 0.12
            self.mains_amp = 0.1
            self.emg_burst_amp = 0.6
            self.emg_burst_count = 2
            self.baseline_shift_amp = 0.0
            self.baseline_shift_count = 0
            self.dropout_count = 1
            self.clip_limit = 4.0
        elif self.difficulty == Difficulty.HARD:
            self.gaussian_amp = 0.2
            self.pink_amp = 0.18
            self.mains_amp = 0.15
            self.emg_burst_amp = 0.9
            self.emg_burst_count = 3
            self.baseline_shift_amp = 0.8
            self.baseline_shift_count = 2
            self.dropout_count = 2
            self.clip_limit = 3.5
        # EXTREME keeps the default values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
    reference = np.asarray(reference, dtype=float)
    test = np.asarray(test, dtype=float)
    noise = reference - test
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def compute_rmse(reference: np.ndarray, test: np.ndarray) -> float:
    """Root Mean Square Error."""
    return np.sqrt(np.mean((reference - test) ** 2))


def compute_mae(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(reference - test))


def compute_pearson(reference: np.ndarray, test: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return np.corrcoef(reference, test)[0, 1]


def compute_psnr(reference: np.ndarray, test: np.ndarray, max_val: Optional[float] = None) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    if max_val is None:
        max_val = np.max(np.abs(reference))
    mse = np.mean((reference - test) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_nrmse(reference: np.ndarray, test: np.ndarray) -> float:
    """Normalized Root Mean Square Error (by range)."""
    return compute_rmse(reference, test) / (np.max(reference) - np.min(reference))


def compute_all_metrics(reference: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    """Compute a dictionary of all metrics."""
    return {
        "SNR (dB)": compute_snr_db(reference, test),
        "RMSE": compute_rmse(reference, test),
        "MAE": compute_mae(reference, test),
        "Pearson": compute_pearson(reference, test),
        "PSNR (dB)": compute_psnr(reference, test),
        "NRMSE": compute_nrmse(reference, test),
    }


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def generate_true_signal(duration_s: float, fs: int, signal_freq: float) -> np.ndarray:
    """Generate clean physiological oscillation (sine wave)."""
    t = np.linspace(0.0, duration_s, int(duration_s * fs), endpoint=False)
    return np.sin(2 * np.pi * signal_freq * t)


def generate_baseline(duration_s: float, fs: int, baseline_freq: float, amplitude: float = 0.5) -> np.ndarray:
    """Generate slow baseline drift."""
    t = np.linspace(0.0, duration_s, int(duration_s * fs), endpoint=False)
    return amplitude * np.sin(2 * np.pi * baseline_freq * t)


# ---------------------------------------------------------------------------
# NoiseGenerator class
# ---------------------------------------------------------------------------
class NoiseGenerator:
    """Composable noise generator for biosignals."""

    def __init__(self, fs: int, rng: Optional[np.random.Generator] = None):
        self.fs = fs
        self.rng = rng if rng is not None else np.random.default_rng()

    def _pink_noise(self, n: int) -> np.ndarray:
        """Generate unit-variance 1/f noise."""
        if n <= 1:
            return np.zeros(n)
        white = self.rng.normal(size=n)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = freqs[1]
        spectrum = spectrum / np.sqrt(freqs)
        pink = np.fft.irfft(spectrum, n)
        std = np.std(pink)
        if std > 0:
            pink = pink / std
        return pink

    def add_gaussian(self, signal: np.ndarray, amplitude: float) -> np.ndarray:
        """Add white Gaussian noise."""
        if amplitude == 0:
            return signal
        return signal + self.rng.normal(scale=amplitude, size=signal.shape)

    def add_pink(self, signal: np.ndarray, amplitude: float) -> np.ndarray:
        """Add pink (1/f) noise."""
        if amplitude == 0:
            return signal
        return signal + amplitude * self._pink_noise(signal.size)

    def add_mains(self, signal: np.ndarray, freq: float, amplitude: float) -> np.ndarray:
        """Add sinusoidal mains interference."""
        if amplitude == 0:
            return signal
        t = np.arange(signal.size) / self.fs
        return signal + amplitude * np.sin(2 * np.pi * freq * t)

    def add_emg_bursts(self, signal: np.ndarray,
                       burst_count: int,
                       burst_duration_s: float,
                       burst_amp: float) -> np.ndarray:
        """Add EMG-like high-frequency bursts."""
        if burst_count == 0 or burst_amp == 0:
            return signal
        n = signal.size
        burst_len = int(burst_duration_s * self.fs)
        if burst_len < 1:
            return signal
        for _ in range(burst_count):
            start = self.rng.integers(0, max(1, n - burst_len))
            envelope = np.hanning(burst_len)
            burst = self.rng.normal(scale=burst_amp, size=burst_len) * envelope
            end = min(start + burst_len, n)
            actual_len = end - start
            signal[start:end] += burst[:actual_len]
        return signal

    def add_baseline_shifts(self, signal: np.ndarray,
                            shift_count: int,
                            shift_len_range_s: Tuple[float, float],
                            shift_amp: float) -> np.ndarray:
        """Add abrupt baseline shifts (rectangular)."""
        if shift_count == 0 or shift_amp == 0:
            return signal
        n = signal.size
        min_s, max_s = shift_len_range_s
        for _ in range(shift_count):
            length = int(self.rng.uniform(min_s, max_s) * self.fs)
            length = max(1, min(length, n))
            start = self.rng.integers(0, max(1, n - length))
            sign = self.rng.choice([-1.0, 1.0])
            signal[start:start + length] += sign * shift_amp
        return signal

    def add_dropouts(self, signal: np.ndarray,
                     dropout_count: int,
                     dropout_len_range_s: Tuple[float, float],
                     flatline_noise_amp: float) -> np.ndarray:
        """Simulate signal dropouts (flatline with small noise)."""
        if dropout_count == 0:
            return signal
        n = signal.size
        min_s, max_s = dropout_len_range_s
        for _ in range(dropout_count):
            length = int(self.rng.uniform(min_s, max_s) * self.fs)
            length = max(1, min(length, n))
            start = self.rng.integers(0, max(1, n - length))
            hold_value = signal[start]
            signal[start:start + length] = (
                hold_value + self.rng.normal(scale=flatline_noise_amp, size=length)
            )
        return signal

    def add_clipping(self, signal: np.ndarray, clip_limit: Optional[float]) -> np.ndarray:
        """Apply sensor clipping (saturation)."""
        if clip_limit is None:
            return signal
        return np.clip(signal, -clip_limit, clip_limit)

    def corrupt(self, clean: np.ndarray, config: PipelineConfig) -> np.ndarray:
        """Apply all noise components according to config."""
        signal = clean.copy()
        if config.gaussian_amp > 0:
            signal = self.add_gaussian(signal, config.gaussian_amp)
        if config.pink_amp > 0:
            signal = self.add_pink(signal, config.pink_amp)
        if config.mains_amp > 0:
            signal = self.add_mains(signal, config.mains_freq, config.mains_amp)
        if config.emg_burst_amp > 0 and config.emg_burst_count > 0:
            signal = self.add_emg_bursts(signal, config.emg_burst_count,
                                         config.emg_burst_duration_s, config.emg_burst_amp)
        if config.baseline_shift_amp > 0 and config.baseline_shift_count > 0:
            signal = self.add_baseline_shifts(signal, config.baseline_shift_count,
                                              config.baseline_shift_len_range_s,
                                              config.baseline_shift_amp)
        if config.dropout_count > 0:
            signal = self.add_dropouts(signal, config.dropout_count,
                                       config.dropout_len_range_s,
                                       config.dropout_flatline_noise_amp)
        if config.clip_limit is not None:
            signal = self.add_clipping(signal, config.clip_limit)
        return signal


# ---------------------------------------------------------------------------
# Pipeline stages (repair, filters)
# ---------------------------------------------------------------------------
def hampel_mask(signal: np.ndarray,
                window_size_sec: float,
                n_sigmas: float,
                fs: int) -> np.ndarray:
    """Hampel outlier mask with window size in seconds."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    n = signal.size
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    # Compute window size in samples, ensure odd
    window = int(window_size_sec * fs)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    half = window // 2

    k = 1.4826  # MAD -> std scale
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        win = signal[left:right]
        med = np.median(win)
        mad = k * np.median(np.abs(win - med))
        if mad == 0:
            continue
        if abs(signal[i] - med) > n_sigmas * mad:
            mask[i] = True
    return mask


def interpolate_mask_adaptive(signal: np.ndarray,
                              mask: np.ndarray,
                              fs: int) -> np.ndarray:
    """
    Replace masked samples with interpolation.
    Short gaps (<40ms) -> PCHIP; medium (40-150ms) -> CubicSpline;
    long (>150ms) -> linear interpolation.
    """
    signal = np.asarray(signal, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if signal.ndim != 1 or mask.shape != signal.shape:
        raise ValueError("signal and mask must be 1-D of same length")
    n = signal.size
    if n == 0 or not mask.any():
        return signal.copy()

    repaired = signal.copy()
    # Find contiguous masked segments
    diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        seg_len = end - start
        if seg_len == 0:
            continue
        # Determine method
        if seg_len / fs <= 0.04:          # <= 40 ms
            method = 'pchip'
        elif seg_len / fs <= 0.15:        # <= 150 ms
            method = 'cubic'
        else:
            method = 'linear'

        # Find good samples around the gap
        # We need at least 2 points on each side for interpolation, but we can use all good.
        good_idx = np.where(~mask)[0]
        if len(good_idx) < 2:
            # Not enough good samples, fallback to linear interpolation with boundaries
            method = 'linear'
        # Use all good points
        x_good = good_idx
        y_good = signal[good_idx]
        if len(x_good) < 2:
            # Cannot interpolate, leave as is (or NaN)
            continue

        if method == 'pchip':
            interp = PchipInterpolator(x_good, y_good, extrapolate=True)
            repaired[start:end] = interp(np.arange(start, end))
        elif method == 'cubic':
            # CubicSpline requires at least 4 points, if not enough fallback to pchip
            if len(x_good) >= 4:
                interp = CubicSpline(x_good, y_good, extrapolate=True)
                repaired[start:end] = interp(np.arange(start, end))
            else:
                interp = PchipInterpolator(x_good, y_good, extrapolate=True)
                repaired[start:end] = interp(np.arange(start, end))
        else:  # linear
            # Use linear interpolation (np.interp works with any number of points)
            repaired[start:end] = np.interp(np.arange(start, end), x_good, y_good)

    return repaired


def notch_filter_sos(x: np.ndarray, fs: int, freq: float, q: float = 30.0) -> np.ndarray:
    """Notch filter using SOS for stability."""
    nyquist = 0.5 * fs
    if not 0 < freq < nyquist:
        raise ValueError(f"freq must be 0 < freq < fs/2")
    # Use iirnotch to get b,a, then convert to sos
    b, a = iirnotch(freq, q, fs)
    sos = signal.tf2sos(b, a)
    return sosfiltfilt(sos, x)


def butter_bandpass_sos(x: np.ndarray,
                        fs: int,
                        low_cutoff: Optional[float],
                        high_cutoff: float,
                        order: int = 4) -> np.ndarray:
    """Butterworth bandpass (or lowpass) using SOS and sosfiltfilt."""
    nyquist = 0.5 * fs
    if not 0 < high_cutoff < nyquist:
        raise ValueError(f"high_cutoff must be 0 < high_cutoff < fs/2")
    if low_cutoff is not None:
        if not 0 < low_cutoff < high_cutoff:
            raise ValueError(f"low_cutoff must be 0 < low_cutoff < high_cutoff")
    if order < 1:
        raise ValueError("order must be >= 1")

    if low_cutoff is None:
        sos = butter(order, high_cutoff / nyquist, btype='low', output='sos')
    else:
        sos = butter(order, [low_cutoff / nyquist, high_cutoff / nyquist],
                     btype='band', output='sos')
    return sosfiltfilt(sos, x)


def wavelet_denoise_bayes(signal: np.ndarray,
                          noise_reference: Optional[np.ndarray] = None,
                          wavelet: str = 'sym8',
                          level: int = 4,
                          mode: str = 'soft') -> np.ndarray:
    """
    Wavelet denoising using BayesShrink threshold.
    If noise_reference is provided, estimate noise sigma from it,
    otherwise estimate from signal's finest detail.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    n = signal.size
    if n == 0:
        return signal.copy()

    # Determine decomposition level
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    if max_level < 1:
        raise ValueError("signal too short for wavelet")
    level = min(level, max_level)

    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    details = coeffs[1:]

    # Estimate noise sigma from reference or from signal's finest detail
    if noise_reference is not None:
        ref_coeffs = pywt.wavedec(noise_reference, wavelet, level=level)
        finest = ref_coeffs[-1]
    else:
        finest = details[-1]
    sigma = np.median(np.abs(finest)) / 0.6745 if finest.size else 0.0

    # BayesShrink: threshold per level = sigma^2 / sigma_x
    # sigma_x estimated from detail coefficients at that level
    denoised_details = []
    for d in details:
        if sigma == 0 or d.size == 0:
            denoised_details.append(d)
            continue
        # Estimate sigma_x from coefficients (robust)
        sigma_x = np.sqrt(np.mean(d ** 2))
        if sigma_x < 1e-12:
            denoised_details.append(d)
            continue
        thr = (sigma ** 2) / sigma_x
        # Apply soft threshold
        denoised_details.append(pywt.threshold(d, value=thr, mode=mode))
    denoised_coeffs = [coeffs[0]] + denoised_details
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    return reconstructed[:n]


# ---------------------------------------------------------------------------
# StageResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    name: str
    signal: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metrics:
            self.metrics = {}


# ---------------------------------------------------------------------------
# Denoising Pipeline class
# ---------------------------------------------------------------------------
class DenoisingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results: List[StageResult] = []
        self.true_signal = None
        self.baseline = None
        self.clean_observed = None
        self.noisy = None

    def run(self) -> None:
        """Execute the full pipeline."""
        cfg = self.config
        fs = cfg.fs

        # Generate true signal and baseline
        self.true_signal = generate_true_signal(cfg.duration_s, fs, cfg.signal_freq)
        self.baseline = generate_baseline(cfg.duration_s, fs, cfg.baseline_freq)
        self.clean_observed = self.true_signal + self.baseline

        # Add noise
        rng = np.random.default_rng(cfg.seed)
        noise_gen = NoiseGenerator(fs, rng)
        self.noisy = noise_gen.corrupt(self.clean_observed, cfg)

        # ---- Pipeline stages ----
        # 1. Hampel mask
        mask = hampel_mask(self.noisy, cfg.hampel_window_sec, cfg.hampel_n_sigmas, fs)
        # 2. Dilation
        dilation_samples = int(cfg.dilation_sec * fs)
        if dilation_samples > 0:
            mask = binary_dilation(mask, iterations=dilation_samples)
        # 3. Adaptive interpolation
        after_interp = interpolate_mask_adaptive(self.noisy, mask, fs)
        # 4. Notch
        after_notch = notch_filter_sos(after_interp, fs, cfg.notch_freq, cfg.notch_q)
        # 5. Butterworth bandpass
        after_butter = butter_bandpass_sos(
            after_notch, fs, cfg.bandpass_low, cfg.bandpass_high, cfg.butter_order
        )
        # 6. Wavelet denoising (using after_interp as noise reference)
        final = wavelet_denoise_bayes(
            after_butter,
            noise_reference=after_interp,
            wavelet=cfg.wavelet_name,
            level=cfg.wavelet_level,
            mode=cfg.wavelet_mode
        )

        # Store results
        self.results = [
            StageResult("Raw noisy", self.noisy),
            StageResult("After repair (Hampel+interp)", after_interp),
            StageResult("After notch", after_notch),
            StageResult("After Butterworth", after_butter),
            StageResult("Final (wavelet)", final),
        ]

        # Compute metrics against true signal (not baseline)
        for res in self.results:
            res.metrics = compute_all_metrics(self.true_signal, res.signal)

    def get_metrics_dataframe(self) -> Dict[str, Dict[str, float]]:
        """Return metrics as nested dict: stage -> metric -> value."""
        return {res.name: res.metrics for res in self.results}

    def print_summary(self) -> None:
        """Print metrics and stage contributions."""
        print("=" * 70)
        print("Denoising Pipeline Summary")
        print(f"Difficulty: {self.config.difficulty.value}")
        print(f"Seed: {self.config.seed}")
        print("-" * 70)
        # Print metrics table
        headers = ["Stage"] + list(self.results[0].metrics.keys())
        col_width = max(len(h) for h in headers) + 2
        print(f"{headers[0]:<{col_width}}", end="")
        for h in headers[1:]:
            print(f"{h:>12}", end="")
        print()
        print("-" * (col_width + 12 * (len(headers)-1)))
        for res in self.results:
            print(f"{res.name:<{col_width}}", end="")
            for key in headers[1:]:
                val = res.metrics.get(key, float('nan'))
                if key in ["SNR (dB)", "PSNR (dB)"]:
                    print(f"{val:>12.2f}", end="")
                else:
                    print(f"{val:>12.4f}", end="")
            print()
        print("-" * 70)

        # Stage contributions (delta SNR)
        snr_values = [res.metrics["SNR (dB)"] for res in self.results]
        print("Stage SNR contributions (delta vs previous):")
        print(f"  {self.results[0].name}: {snr_values[0]:.2f} dB")
        for i in range(1, len(self.results)):
            delta = snr_values[i] - snr_values[i-1]
            print(f"  {self.results[i].name}: {snr_values[i]:.2f} dB  (Δ = {delta:+.2f} dB)")
        total_imp = snr_values[-1] - snr_values[0]
        print(f"Total improvement: {total_imp:+.2f} dB")
        print("=" * 70)

    def plot(self, save_path: Optional[str] = None) -> None:
        """Plot time-domain signals, FFT, and PSD for each stage."""
        cfg = self.config
        t = np.arange(self.noisy.size) / cfg.fs

        # Figure layout: 5 rows (stages) x 3 columns (time, FFT, PSD)
        fig, axes = plt.subplots(len(self.results), 3, figsize=(15, 3*len(self.results)))
        fig.suptitle(f"Denoising pipeline - Difficulty: {cfg.difficulty.value}", fontsize=14)

        for i, res in enumerate(self.results):
            signal_data = res.signal
            # Time domain
            ax_time = axes[i, 0]
            ax_time.plot(t, signal_data, lw=0.8, color=f'C{i}')
            if i == 0:
                ax_time.set_title("Time domain")
            ax_time.set_ylabel("Amplitude")
            ax_time.grid(alpha=0.3)
            if i == len(self.results)-1:
                ax_time.set_xlabel("Time (s)")

            # FFT magnitude
            ax_fft = axes[i, 1]
            fft_vals = np.fft.rfft(signal_data)
            freqs = np.fft.rfftfreq(signal_data.size, 1/cfg.fs)
            ax_fft.plot(freqs, np.abs(fft_vals), lw=0.8, color=f'C{i}')
            ax_fft.set_xlim(0, cfg.fs/2)
            if i == 0:
                ax_fft.set_title("FFT magnitude")
            ax_fft.set_ylabel("Magnitude")
            ax_fft.grid(alpha=0.3)
            if i == len(self.results)-1:
                ax_fft.set_xlabel("Frequency (Hz)")

            # PSD (Welch)
            ax_psd = axes[i, 2]
            f, Pxx = signal.welch(signal_data, cfg.fs, nperseg=min(256, signal_data.size//2))
            ax_psd.semilogy(f, Pxx, lw=0.8, color=f'C{i}')
            ax_psd.set_xlim(0, cfg.fs/2)
            if i == 0:
                ax_psd.set_title("PSD (Welch)")
            ax_psd.set_ylabel("Power")
            ax_psd.grid(alpha=0.3)
            if i == len(self.results)-1:
                ax_psd.set_xlabel("Frequency (Hz)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        plt.show()

    def save_results(self, path_prefix: str) -> None:
        """Save stage signals and metrics to disk."""
        np.savez(f"{path_prefix}_signals.npz",
                 true_signal=self.true_signal,
                 baseline=self.baseline,
                 clean_observed=self.clean_observed,
                 noisy=self.noisy,
                 **{res.name.replace(" ", "_"): res.signal for res in self.results})
        # Save metrics as CSV-like text
        with open(f"{path_prefix}_metrics.txt", "w") as f:
            f.write("Stage, " + ", ".join(self.results[0].metrics.keys()) + "\n")
            for res in self.results:
                vals = [str(v) for v in res.metrics.values()]
                f.write(f"{res.name}, " + ", ".join(vals) + "\n")


# ---------------------------------------------------------------------------
# Main entry point (demo)
# ---------------------------------------------------------------------------
def main():
    # Create configuration (adjust difficulty here)
    config = PipelineConfig(
        duration_s=10.0,
        fs=250,
        difficulty=Difficulty.EXTREME,  # change to EASY, MEDIUM, HARD, EXTREME
        seed=42,
        n_repeats=1  # for simplicity we run once, but could loop
    )

    # Optionally run multiple repeats and average metrics
    if config.n_repeats > 1:
        all_metrics = []
        for seed_offset in range(config.n_repeats):
            cfg = PipelineConfig(**{**config.__dict__, "seed": config.seed + seed_offset})
            pipe = DenoisingPipeline(cfg)
            pipe.run()
            all_metrics.append(pipe.get_metrics_dataframe())
        # Average metrics across repeats
        avg_metrics = {}
        for stage_name in all_metrics[0].keys():
            avg_metrics[stage_name] = {}
            for metric in all_metrics[0][stage_name].keys():
                vals = [m[stage_name][metric] for m in all_metrics]
                avg_metrics[stage_name][metric] = np.mean(vals)
        print("Average metrics over {} repeats:".format(config.n_repeats))
        for stage, mets in avg_metrics.items():
            print(f"  {stage}: {mets}")
    else:
        pipe = DenoisingPipeline(config)
        pipe.run()
        pipe.print_summary()
        pipe.plot(save_path="denoise_demo_improved.png")
        pipe.save_results("denoise_demo_results")


if __name__ == "__main__":
    main()