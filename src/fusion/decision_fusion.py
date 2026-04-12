def fuse_predictions(cnn_score, hr_quality_score, micro_score, hr_bpm=None):
    """
    CNN-dominant fusion (robust to micro errors)
    """

    # ---- safety ----
    if hr_quality_score is None:
        hr_quality_score = 0.0

    if micro_score is None:
        micro_score = 0.5

    # ---- NEW WEIGHTS ----
    w_cnn = 0.5     # 🔥 main signal
    w_micro = 0.3
    w_hr = 0.2

    # ---- BASE SCORE ----
    final_score = (
        w_cnn * cnn_score +
        w_micro * micro_score +
        w_hr * hr_quality_score
    )

    # ---- RULES (VERY IMPORTANT) ----

    # Strong FAKE from CNN (trust CNN)
    if cnn_score < 0.3:
        final_score *= 0.6

    # Strong REAL from CNN
    if cnn_score > 0.8:
        final_score = max(final_score, 0.75)

    # Micro disagreement correction
    if micro_score > 0.8 and cnn_score < 0.5:
        final_score *= 0.7  # reduce false real

    if micro_score < 0.2 and cnn_score > 0.5:
        final_score *= 0.85  # slight correction

    # HR sanity
    if hr_bpm is not None and (hr_bpm < 50 or hr_bpm > 120):
        final_score *= 0.85

    # ---- SHARPEN ----
    final_score = final_score ** 1.1

    # ---- DECISION ----
    THRESHOLD = 0.6
    final_label = "REAL" if final_score > THRESHOLD else "FAKE"

    # ---- CONFIDENCE ----
    confidence = abs(final_score - 0.5) * 2
    confidence = min(confidence, 1.0)

    # Boost confidence when CNN is strong
    if cnn_score > 0.8 or cnn_score < 0.2:
        confidence = min(confidence + 0.2, 1.0)

    return {
        "final_label": final_label,
        "final_score": float(final_score),
        "confidence": float(confidence),

        "cnn_score": float(cnn_score),
        "hr_quality_score": float(hr_quality_score),
        "micro_expression_score": float(micro_score)
    }