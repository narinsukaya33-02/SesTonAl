from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from analyzer import analyze_voice_sample
from report_generator import generate_pdf_report
from scoring import compute_respiratory_risk

st.set_page_config(
    page_title="SesTonAI · Akustik Solunum Risk Tarama",
    page_icon="🎙️",
    layout="wide",
)

st.title("SesTonAI — Akustik Vokal Biyobelirteç & Solunum Risk Karar Destek Sistemi")
st.caption(
    "Deneysel karar destek prototipidir; klinik tanı veya tedavi yerine geçmez. "
    "Lütfen 3–15 saniyelik net bir ses kaydı (.wav, .flac, .ogg, .mp3, .m4a) yükleyin."
)

with st.sidebar:
    st.header("Tarama Bağlamı (Opsiyonel)")
    participant_label = st.text_input("Kayıt / Katılımcı Etiketi", value="Örnek Kayıt #01")

    st.subheader("Eşlik Eden Durumlar")
    shortness_of_breath = st.checkbox("Nefes darlığı / eforla çabuk yorulma", value=False)
    cough_or_wheeze = st.checkbox("Öksürük veya hırıltılı solunum", value=False)
    fatigue_or_fever = st.checkbox("Halsizlik veya ateş", value=False)
    smoker_or_irritant = st.checkbox("Sigara veya duman/tahriş edici maruziyeti", value=False)

    st.divider()
    st.info(
        "Ses kaydı tarayıcıdan yüklendikten sonra akustik özellikler (F0, Jitter, Shimmer, HNR, "
        "duraksama kadansı ve spektral dağılım) otomatik hesaplanır."
    )

uploaded_file = st.file_uploader(
    "Ses kaydı yükleyin",
    type=["wav", "flac", "ogg", "mp3", "m4a"],
    help="Net ve arka plan gürültüsü düşük bir kayıt daha kararlı sonuç verir.",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    st.audio(file_bytes, format=uploaded_file.type or "audio/wav")

    try:
        with st.spinner("Ses sinyali işleniyor ve akustik biyobelirteçler çıkarılıyor..."):
            features = analyze_voice_sample(file_bytes, filename=uploaded_file.name)
            context = {
                "shortness_of_breath": shortness_of_breath,
                "cough_or_wheeze": cough_or_wheeze,
                "fatigue_or_fever": fatigue_or_fever,
                "smoker_or_irritant": smoker_or_irritant,
            }
            scoring = compute_respiratory_risk(features, context=context)
    except Exception as exc:
        st.error(f"Ses analizi sırasında bir hata oluştu: {exc}")
        st.stop()

    st.divider()
    st.subheader("1. Karar Destek Risk Özeti")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Skoru", f"{scoring['risk_score']} / 100")
    c2.metric("Risk Seviyesi", scoring["risk_level"])
    c3.metric("Kayıt Süresi", f"{features['duration_sec']} sn")
    c4.metric("Duraksama Sayısı", f"{features['pause_events']} ({features['pause_ratio']:.0%})")

    if scoring["risk_level"] == "Yüksek":
        st.warning(scoring["summary"])
    else:
        st.info(scoring["summary"])

    st.subheader("2. Akustik Biyobelirteç Metrikleri")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temel Frekans (F0)", f"{features['f0_mean_hz']} Hz", help="Sesli bölümlerde ortalama temel frekans")
    m2.metric("Jitter (Local)", f"{features['jitter_local_pct']}%")
    m3.metric("Shimmer (Local)", f"{features['shimmer_local_pct']}%")
    m4.metric("Harmonik-Gürültü (HNR)", f"{features['hnr_db']} dB")

    with st.expander("Detaylı Spektral & MFCC Değerleri"):
        clean_metrics = {k: v for k, v in features.items() if not k.startswith("_")}
        st.json(clean_metrics)

    st.subheader("3. Dalga Formu ve Spektrogram")
    g1, g2 = st.columns(2)

    with g1:
        fig_wave, ax_wave = plt.subplots(figsize=(6, 2.8))
        wave = np.asarray(features["_waveform"], dtype=np.float32)
        t_axis = np.linspace(0.0, float(features["duration_sec"]), num=len(wave))
        ax_wave.plot(t_axis, wave, linewidth=0.8)
        ax_wave.set_title("Normalize Dalga Formu")
        ax_wave.set_xlabel("Zaman (sn)")
        ax_wave.set_ylabel("Genlik")
        ax_wave.grid(True, alpha=0.25)
        fig_wave.tight_layout()
        st.pyplot(fig_wave)
        plt.close(fig_wave)

    with g2:
        fig_spec, ax_spec = plt.subplots(figsize=(6, 2.8))
        spec_db = np.asarray(features["_spectrogram_db"], dtype=np.float32)
        img = ax_spec.imshow(
            spec_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0.0, float(features["duration_sec"]), 0.0, float(features["sample_rate"]) / 2000.0],
        )
        ax_spec.set_title("Spektrogram (dB)")
        ax_spec.set_xlabel("Zaman (sn)")
        ax_spec.set_ylabel("Frekans (kHz)")
        fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
        fig_spec.tight_layout()
        st.pyplot(fig_spec)
        plt.close(fig_spec)

    st.subheader("4. Skor Katkı Dağılımı ve Rapor İndirme")
    contrib_data = {
        item["factor"]: item["points"] for item in scoring.get("contributions", [])
    }
    st.bar_chart(contrib_data)

    d1, d2 = st.columns(2)
    pdf_bytes = generate_pdf_report(features, scoring, participant_label=participant_label)
    with d1:
        st.download_button(
            label="📄 PDF Raporu İndir",
            data=pdf_bytes,
            file_name="sestonai_rapor.pdf",
            mime="application/pdf",
        )

    export_payload = {
        "participant_label": participant_label,
        "scoring": scoring,
        "features": {k: v for k, v in features.items() if not k.startswith("_")},
    }
    with d2:
        st.download_button(
            label="🧾 JSON Verisini İndir",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="sestonai_analiz.json",
            mime="application/json",
        )
else:
    st.info("Analize başlamak için sol veya üst alandan bir ses dosyası yükleyin.")
