from __future__ import annotations

from datetime import datetime, timezone
import io
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def _safe_pdf_text(text: str) -> str:
    """Standart Type1 PDF fontlarında bozulma olmaması için Türkçe karakterleri güvenli karşılıklarına çevirir."""
    replacements = {
        "ğ": "g",
        "Ğ": "G",
        "ü": "u",
        "Ü": "U",
        "ş": "s",
        "Ş": "S",
        "ı": "i",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ç": "c",
        "Ç": "C",
        "—": "-",
        "–": "-",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.encode("latin-1", errors="replace").decode("latin-1")


def _wrap_lines(text: str, max_chars: int = 86) -> list[str]:
    words = _safe_pdf_text(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        if len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def generate_pdf_report(
    features: dict[str, Any],
    scoring: dict[str, Any],
    participant_label: str = "Anonim Katilimci",
) -> bytes:
    """Analiz sonuçlarından çok satırlı, taşma yapmayan bir PDF özet raporu üretir."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _, height = A4

    margin_x = 20 * mm
    y = height - 22 * mm
    line_h = 5.2 * mm

    def write_line(text: str, font: str = "Helvetica", size: int = 10, gap_after: float = 1.5) -> None:
        nonlocal y
        if y < 22 * mm:
            c.showPage()
            y = height - 22 * mm
        c.setFont(font, size)
        for wrapped in _wrap_lines(text, max_chars=88 if size <= 10 else 68):
            if y < 22 * mm:
                c.showPage()
                y = height - 22 * mm
                c.setFont(font, size)
            c.drawString(margin_x, y, wrapped)
            y -= line_h
        y -= gap_after * mm

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    write_line("SesTonAI - Akustik Vokalle Solunum Risk Tarama Karar Destek Raporu", font="Helvetica-Bold", size=13, gap_after=2.0)
    write_line(f"Olusturulma Zamani: {timestamp} | Kayit Etiketi: {participant_label}", font="Helvetica", size=9, gap_after=3.0)

    write_line("1. Risk Degerlendirme Ozeti", font="Helvetica-Bold", size=11, gap_after=1.0)
    write_line(f"Risk Skoru: {scoring.get('risk_score', 0.0)} / 100  |  Risk Seviyesi: {scoring.get('risk_level', 'Bilinmiyor')}", font="Helvetica-Bold", size=10, gap_after=1.0)
    write_line(f"Degerlendirme: {scoring.get('summary', '')}", font="Helvetica", size=10, gap_after=3.0)

    write_line("2. Cikarilan Akustik Biyobelirtec Metrikleri", font="Helvetica-Bold", size=11, gap_after=1.0)
    metric_rows = [
        f"Dosya Adi: {features.get('filename', '-')} | Sure: {features.get('duration_sec', 0.0)} sn | Ornekleme: {features.get('sample_rate', 16000)} Hz",
        f"Temel Frekans (F0): {features.get('f0_mean_hz', 0.0)} Hz (Std: {features.get('f0_std_hz', 0.0)} Hz) | Sesli Oran: {features.get('voiced_fraction', 0.0)}",
        f"Jitter (Local): {features.get('jitter_local_pct', 0.0)}% | Shimmer (Local): {features.get('shimmer_local_pct', 0.0)}% | HNR: {features.get('hnr_db', 0.0)} dB",
        f"Duraksama Orani: {features.get('pause_ratio', 0.0)} | Duraksama Sayisi: {features.get('pause_events', 0)} | Yuksek Frekans Orani: {features.get('high_freq_ratio', 0.0)}",
        f"Spektral Merkez: {features.get('spectral_centroid_hz', 0.0)} Hz | Spektral Bant Genisligi: {features.get('spectral_bandwidth_hz', 0.0)} Hz",
    ]
    for row in metric_rows:
        write_line(f"- {row}", font="Helvetica", size=9.5, gap_after=0.8)

    y -= 2 * mm
    write_line("3. Skor Katki Dagilimi", font="Helvetica-Bold", size=11, gap_after=1.0)
    for item in scoring.get("contributions", []):
        write_line(
            f"- {item.get('factor', '')}: {item.get('points', 0.0)} / {item.get('max_points', 0.0)} puan",
            font="Helvetica",
            size=9.5,
            gap_after=0.8,
        )

    y -= 2 * mm
    write_line("4. Klinik Uyari", font="Helvetica-Bold", size=11, gap_after=1.0)
    write_line(
        scoring.get("disclaimer", "Bu rapor deneysel karar destek amaclidir; tibbi tani yerine gecmez."),
        font="Helvetica-Oblique",
        size=9,
        gap_after=1.0,
    )

    c.save()
    buf.seek(0)
    return buf.getvalue()
