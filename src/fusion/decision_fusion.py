def fuse_decisions(cnn_score, hr_quality_score, micro_expression_score, thresholds=None):
    """
    Combines deepfake CNN probability + physiological heart-rate stability
    into a final REAL/FAKE decision.

    Args:
        cnn_score (float): Probability from CNN (0–1), where 1 = fake
        hr_quality_score (float): rPPG reliability (0–1), where 1 = stable/real
        thresholds (dict): custom thresholds (optional)

    Returns:
        dict with final fusion output
    """

    default_thresholds = {
        "cnn_fake_threshold": 0.55,
        "hr_min_real_threshold": 0.35,
        "microexp_min_real_threshold": 0.30,
    }

    if thresholds is None:
        thresholds = default_thresholds

    cnn_th = thresholds["cnn_fake_threshold"]
    hr_th = thresholds["hr_min_real_threshold"]
    micro_th = thresholds["microexp_min_real_threshold"]

    # Case 1 — Strong FAKE from CNN + bad HR signal
    if cnn_score >= cnn_th and hr_quality_score < 0.5:
        label = "FAKE"
        confidence = (cnn_score * 0.7) + ((1 - hr_quality_score) * 0.3)
        return {
            "final_label": label,
            "cnn_score": cnn_score,
            "hr_quality_score": hr_quality_score,
            "micro_expression_score": micro_expression_score,
            "confidence": float(confidence)
        }

    # Case 2 — HR unstable → suspicious
    if hr_quality_score < hr_th:
        if cnn_score >= 0.40:
            label = "FAKE"
            confidence = (cnn_score * 0.5) + ((1 - hr_quality_score) * 0.5)
        else:
            label = "FAKE"
            confidence = 0.55

        return {
            "final_label": label,
            "cnn_score": cnn_score,
            "hr_quality_score": hr_quality_score,
            "micro_expression_score": micro_expression_score,
            "confidence": float(confidence)
        }

    # Case 3 — Micro-expression analysis (if available)
    if micro_expression_score < micro_th and cnn_score >= 0.45:
        label = "FAKE"
        confidence = (cnn_score * 0.5) + ((1 - micro_expression_score) * 0.5)

        return {
            "final_label": label,
            "cnn_score": cnn_score,
            "hr_quality_score": hr_quality_score,
            "micro_expression_score": micro_expression_score,
            "confidence": float(confidence)
        }

    # Case 3 — All indicators support REAL
    label = "REAL"
    confidence = (
    (1 - cnn_score) * 0.4 +
    hr_quality_score * 0.3 +
    micro_expression_score * 0.3
    )

    return {
        "final_label": label,
        "cnn_score": cnn_score,
        "hr_quality_score": hr_quality_score,
        "micro_expression_score": micro_expression_score,
        "confidence": float(confidence)
    }
