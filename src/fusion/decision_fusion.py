def fuse_predictions(cnn_score, hr_quality_score, micro_expression_score):

    # weighted fusion
    final_score = (
        0.6 * cnn_score +
        0.2 * hr_quality_score +
        0.2 * micro_expression_score
    )

    # final label
    final_label = "REAL" if final_score > 0.5 else "FAKE"

    # confidence (distance from threshold)
    confidence = abs(final_score - 0.5) * 2

    return {
        "final_label": final_label,
        "cnn_score": cnn_score,
        "hr_quality_score": hr_quality_score,
        "micro_expression_score": micro_expression_score,
        "confidence": confidence
    }