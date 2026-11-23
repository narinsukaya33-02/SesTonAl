from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal
import soundfile as sf


def _read_audio_bytes(file_bytes: bytes, filename: str = "audio.wav") -> tuple[np.ndarray, int]:
    """Ses baytlarını mono float32 dizisine ve örnekleme hızına çevirir."""
    ext = Path(filename).suffix.lower() if filename else ".wav"

    # Önce soundfile ile doğrudan dene
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, int(sr)
    except Exception:
        pass

    # Geçici dosyaya yazıp librosa ile dene (mp3, m4a, ogg vb. için)
    suffix = ext if ext else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        audio, sr = librosa.load(tmp.name, sr=None, mono=True)
        return np.asarray(audio, dtype=np.float32), int(sr)


def _preprocess_audio(audio: np.ndarray, sr: int, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """DC offset temizler, yeniden örnekler, 80-7500 Hz bant geçiren filtre uygular ve normalize eder."""
    if audio.size == 0:
        raise ValueError("Ses dosyası boş veya okunamıyor.")

    # NaN / Inf temizliği
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # DC offset kaldır
    audio = audio - float(np.mean(audio))

    # Hedef örnekleme hızına getir
    if sr != target_sr and sr > 0:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr

    # Hafif Butterworth bant geçiren filtre (80 Hz - 7500 Hz)
    nyquist = 0.5 * sr
    low = max(20.0, 80.0) / nyquist
    high = min(7500.0, nyquist * 0.95) / nyquist
    if 0.0 < low < high < 1.0:
        sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
        audio = signal.sosfiltfilt(sos, audio).astype(np.float32)

    # Tepe normalizasyonu
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = (audio / peak * 0.92).astype(np.float32)

    return audio, sr


def _extract_praat_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Parselmouth üzerinden F0, jitter, shimmer ve HNR metriklerini çıkarır."""
    defaults = {
        "f0_mean_hz": 0.0,
        "f0_std_hz": 0.0,
        "jitter_local_pct": 0.0,
        "shimmer_local_pct": 0.0,
        "hnr_db": 0.0,
        "voiced_fraction": 0.0,
    }
    try:
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=65.0, pitch_ceiling=550.0)
        f0_values = pitch.selected_array["frequency"]
        voiced_mask = f0_values > 0
        voiced_fraction = float(np.mean(voiced_mask)) if f0_values.size > 0 else 0.0

        f0_voiced = f0_values[voiced_mask]
        f0_mean = float(np.mean(f0_voiced)) if f0_voiced.size > 0 else 0.0
        f0_std = float(np.std(f0_voiced)) if f0_voiced.size > 1 else 0.0

        point_process = call(snd, "To PointProcess (periodic, cc)", 65.0, 550.0)
        num_points = call(point_process, "Get number of points")

        jitter_pct = 0.0
        shimmer_pct = 0.0
        if num_points >= 3:
            j_local = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            if j_local is not None and np.isfinite(j_local):
                jitter_pct = float(j_local * 100.0)

            s_local = call(
                [snd, point_process],
                "Get shimmer (local)",
                0.0,
                0.0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            if s_local is not None and np.isfinite(s_local):
                shimmer_pct = float(s_local * 100.0)

        harmonicity = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=65.0, silence_threshold=0.1, periods_per_window=4.5)
        hnr_vals = harmonicity.values[0]
        hnr_valid = hnr_vals[np.isfinite(hnr_vals) & (hnr_vals > -40.0)]
        hnr_db = float(np.mean(hnr_valid)) if hnr_valid.size > 0 else 0.0

        return {
            "f0_mean_hz": round(f0_mean, 2),
            "f0_std_hz": round(f0_std, 2),
            "jitter_local_pct": round(jitter_pct, 3),
            "shimmer_local_pct": round(shimmer_pct, 3),
            "hnr_db": round(hnr_db, 2),
            "voiced_fraction": round(voiced_fraction, 3),
        }
    except Exception:
        return defaults


def _extract_spectral_and_respiratory_features(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Spektral, MFCC ve enerji/duraksama tabanlı solunum akustik özelliklerini hesaplar."""
    duration_sec = float(len(audio) / sr) if sr > 0 else 0.0

    # Kısa zamanlı RMS enerjisi ve nefes/duraksama analizi
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-7), ref=np.max)

    # Düşük enerjili segmentler (duraksama / nefes geçişi adayı)
    pause_mask = rms_db < -28.0
    pause_ratio = float(np.mean(pause_mask)) if pause_mask.size > 0 else 0.0

    # Duraksama sayısı (ardışık düşük enerjili kare grupları >= 120 ms)
    min_pause_frames = max(1, int(0.12 / 0.010))
    pause_events = 0
    run_len = 0
    for is_pause in pause_mask:
        if is_pause:
            run_len += 1
        else:
            if run_len >= min_pause_frames:
                pause_events += 1
            run_len = 0
    if run_len >= min_pause_frames:
        pause_events += 1

    # Spektral centroid, bandwidth, rolloff ve ZCR
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]))
    spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length, roll_percent=0.85)[0]))
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]))

    # Yüksek frekans enerji oranı (> 3000 Hz)
    stft_mag = np.abs(librosa.stft(audio, n_fft=512, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    total_energy = float(np.sum(stft_mag ** 2)) + 1e-9
    high_freq_energy = float(np.sum(stft_mag[freqs >= 3000.0, :] ** 2))
    high_freq_ratio = high_freq_energy / total_energy

    # MFCC ortalamaları (ilk 6 katsayı)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_means = [round(float(np.mean(mfcc[i])), 2) for i in range(6)]

    # Spektrogram ve dalga formu için alt örneklenmiş diziler
    spectrogram_db = librosa.amplitude_to_db(stft_mag, ref=np.max)

    return {
        "duration_sec": round(duration_sec, 2),
        "sample_rate": sr,
        "rms_mean_db": round(float(np.mean(rms_db)), 2),
        "rms_std_db": round(float(np.std(rms_db)), 2),
        "pause_ratio": round(pause_ratio, 3),
        "pause_events": int(pause_events),
        "spectral_centroid_hz": round(spec_centroid, 1),
        "spectral_bandwidth_hz": round(spec_bandwidth, 1),
        "spectral_rolloff_hz": round(spec_rolloff, 1),
        "zcr_mean": round(zcr_mean, 4),
        "high_freq_ratio": round(high_freq_ratio, 4),
        "mfcc_1_to_6": mfcc_means,
        "_waveform": audio[:: max(1, len(audio) // 4000)],
        "_spectrogram_db": spectrogram_db[:, :: max(1, spectrogram_db.shape[1] // 180)],
    }


def analyze_voice_sample(file_bytes: bytes, filename: str = "sample.wav") -> dict[str, Any]:
    """Ses dosyasını analiz eder ve yapılandırılmış akustik biyobelirteç sözlüğü döndürür."""
    raw_audio, raw_sr = _read_audio_bytes(file_bytes, filename=filename)
    audio, sr = _preprocess_audio(raw_audio, raw_sr, target_sr=16000)

    praat_feats = _extract_praat_features(audio, sr)
    spec_feats = _extract_spectral_and_respiratory_features(audio, sr)

    features: dict[str, Any] = {
        "filename": filename,
        **spec_feats,
        **praat_feats,
    }
    return features
