from __future__ import annotations

from typing import Any


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def _norm_linear(value: float, healthy_ref: float, elevated_ref: float) -> float:
    """Değeri 0.0 (normal) ile 1.0 (yüksek sapma) aralığına ölçekler."""
    if elevated_ref == healthy_ref:
        return 0.0
    return _clamp((value - healthy_ref) / (elevated_ref - healthy_ref), 0.0, 1.0)


def compute_respiratory_risk(
    features: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Akustik biyobelirteçlerden deneysel karar destek risk skoru (0-100) hesaplar.
    Not: Klinik tanı yerine geçmez; ön tarama ve takip amaçlı karar destek göstergesidir.
    """
    context = context or {}

    jitter = float(features.get("jitter_local_pct", 0.0))
    shimmer = float(features.get("shimmer_local_pct", 0.0))
    hnr = float(features.get("hnr_db", 0.0))
    pause_ratio = float(features.get("pause_ratio", 0.0))
    pause_events = int(features.get("pause_events", 0))
    duration_sec = max(float(features.get("duration_sec", 1.0)), 0.5)
    high_freq_ratio = float(features.get("high_freq_ratio", 0.0))
    spec_centroid = float(features.get("spectral_centroid_hz", 0.0))

    # 1. Vokal periyodisite düzensizliği (Jitter + Shimmer) -> max 28 puan
    jitter_norm = _norm_linear(jitter, healthy_ref=0.65, elevated_ref=2.50)
    shimmer_norm = _norm_linear(shimmer, healthy_ref=3.00, elevated_ref=10.50)
    vocal_instability_score = (0.55 * jitter_norm + 0.45 * shimmer_norm) * 28.0

    # 2. Harmonik / gürültü oranı (düşük HNR hava kaçağı / glottik kapanma düzensizliği) -> max 22 puan
    if hnr <= 0.0:
        hnr_norm = 0.45
    else:
        hnr_norm = _clamp((18.0 - hnr) / (18.0 - 5.0), 0.0, 1.0)
    hnr_score = hnr_norm * 22.0

    # 3. Solunum duraksama yoğunluğu ve nefes kesilmesi -> max 25 puan
    pauses_per_min = (pause_events / duration_sec) * 60.0
    pause_ratio_norm = _norm_linear(pause_ratio, healthy_ref=0.18, elevated_ref=0.55)
    pause_rate_norm = _norm_linear(pauses_per_min, healthy_ref=6.0, elevated_ref=22.0)
    respiratory_cadence_score = (0.60 * pause_ratio_norm + 0.40 * pause_rate_norm) * 25.0

    # 4. Yüksek frekans türbülansı / spektral kayma -> max 15 puan
    hf_norm = _norm_linear(high_freq_ratio, healthy_ref=0.06, elevated_ref=0.28)
    centroid_norm = _norm_linear(spec_centroid, healthy_ref=1400.0, elevated_ref=3400.0)
    spectral_score = (0.65 * hf_norm + 0.35 * centroid_norm) * 15.0

    # 5. Opsiyonel bağlamsal semptom/risk faktörleri -> max 10 puan
    symptom_flags = [
        bool(context.get("shortness_of_breath", False)),
        bool(context.get("cough_or_wheeze", False)),
        bool(context.get("fatigue_or_fever", False)),
        bool(context.get("smoker_or_irritant", False)),
    ]
    context_score = (sum(symptom_flags) / 4.0) * 10.0

    raw_total = (
        vocal_instability_score
        + hnr_score
        + respiratory_cadence_score
        + spectral_score
        + context_score
    )
    risk_score = round(_clamp(raw_total, 0.0, 100.0), 1)

    if risk_score < 35.0:
        risk_level = "Düşük"
        summary = (
            "Akustik vokal ve solunum duraksama metrikleri genel olarak referans aralığına yakın seyretmektedir. "
            "Belirti devam ederse veya kötüleşirse tıbbi değerlendirme önerilir."
        )
    elif risk_score < 65.0:
        risk_level = "Orta"
        summary = (
            "Ses titreşimi, harmonik kararlılık veya konuşma içi duraksama ritminde orta düzey sapmalar saptanmıştır. "
            "Semptomlarla birlikte değerlendirilmesi ve takip kaydı alınması önerilir."
        )
    else:
        risk_level = "Yüksek"
        summary = (
            "Akustik biyobelirteçlerde belirgin düzensizlik ve/veya sık solunum duraksaması gözlemlenmiştir. "
            "Bu sonuç tanı koymaz; en kısa sürede bir sağlık profesyoneline danışmanız önerilir."
        )

    contributions = [
        {
            "factor": "Vokal Titreşim Kararlılığı (Jitter & Shimmer)",
            "points": round(vocal_instability_score, 1),
            "max_points": 28.0,
        },
        {
            "factor": "Harmonik-Gürültü Oranı (HNR)",
            "points": round(hnr_score, 1),
            "max_points": 22.0,
        },
        {
            "factor": "Solunum Duraksama & Kadans",
            "points": round(respiratory_cadence_score, 1),
            "max_points": 25.0,
        },
        {
            "factor": "Yüksek Frekans & Spektral Yapı",
            "points": round(spectral_score, 1),
            "max_points": 15.0,
        },
        {
            "factor": "Kullanıcı Bağlam / Belirti Girdisi",
            "points": round(context_score, 1),
            "max_points": 10.0,
        },
    ]

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "summary": summary,
        "contributions": contributions,
        "disclaimer": (
            "SesTonAI bir klinik tanı cihazı değildir; deneysel akustik biyobelirteç karar destek sistemidir."
        ),
    }
