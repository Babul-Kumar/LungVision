"""
LungVision AI — Uncertainty Quantification & Clinical Abstention Framework.

Evaluates predictive margin to decision threshold, classifies prediction certainty,
and issues structured abstention or clinician review recommendations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    BORDERLINE = "BORDERLINE"


class ClinicalRecommendation(str, Enum):
    STANDARD_READING = "STANDARD_READING"
    EXPERT_REVIEW_RECOMMENDED = "EXPERT_REVIEW_RECOMMENDED"
    HIGH_ACUITY_ALERT = "HIGH_ACUITY_ALERT"
    TECHNICAL_REPEAT_RECOMMENDED = "TECHNICAL_REPEAT_RECOMMENDED"


@dataclass
class DiseasePredictionDetail:
    disease: str
    probability: float
    threshold: float
    is_positive: bool
    margin: float
    confidence_level: ConfidenceLevel
    clinical_note: str

    def to_dict(self) -> Dict:
        return {
            "disease": self.disease,
            "probability": round(self.probability, 4),
            "threshold": round(self.threshold, 4),
            "is_positive": self.is_positive,
            "margin": round(self.margin, 4),
            "confidence_level": self.confidence_level.value,
            "clinical_note": self.clinical_note
        }


@dataclass
class UncertaintySummary:
    overall_confidence: ConfidenceLevel
    mean_margin: float
    borderline_diseases: List[str]
    positive_findings: List[str]
    recommendation: ClinicalRecommendation
    recommendation_reason: str

    def to_dict(self) -> Dict:
        return {
            "overall_confidence": self.overall_confidence.value,
            "mean_margin": round(self.mean_margin, 4),
            "borderline_diseases": self.borderline_diseases,
            "positive_findings": self.positive_findings,
            "recommendation": self.recommendation.value,
            "recommendation_reason": self.recommendation_reason
        }


class UncertaintyAnalyzer:
    """Analyzes prediction distributions relative to decision thresholds."""

    # Decision margin boundaries
    BORDERLINE_MARGIN: float = 0.05
    LOW_MARGIN: float = 0.15
    MODERATE_MARGIN: float = 0.30

    # High-acuity critical conditions
    HIGH_ACUITY_DISEASES = {"Pneumothorax", "Edema", "Cardiomegaly"}

    def assess_disease(self, disease: str, prob: float, threshold: float) -> DiseasePredictionDetail:
        """Evaluates certainty for a single pathology prediction."""
        prob = float(prob)
        threshold = float(threshold)
        is_pos = prob >= threshold
        margin = abs(prob - threshold)

        if margin < self.BORDERLINE_MARGIN:
            conf = ConfidenceLevel.BORDERLINE
            note = f"Borderline prediction (|p - τ| = {margin:.3f}); near decision boundary."
        elif margin < self.LOW_MARGIN:
            conf = ConfidenceLevel.LOW
            note = f"Low margin above/below threshold; clinical correlation advised."
        elif margin < self.MODERATE_MARGIN:
            conf = ConfidenceLevel.MODERATE
            note = f"Moderate diagnostic confidence."
        else:
            conf = ConfidenceLevel.HIGH
            note = f"Definitive prediction with high boundary separation."

        return DiseasePredictionDetail(
            disease=disease,
            probability=prob,
            threshold=threshold,
            is_positive=is_pos,
            margin=margin,
            confidence_level=conf,
            clinical_note=note
        )

    def summarize(
        self,
        details: List[DiseasePredictionDetail],
        has_quality_warnings: bool = False
    ) -> UncertaintySummary:
        """Synthesizes per-disease assessments into an overall clinical uncertainty summary."""
        margins = [d.margin for d in details]
        mean_margin = float(sum(margins) / max(len(margins), 1))

        borderlines = [d.disease for d in details if d.confidence_level == ConfidenceLevel.BORDERLINE]
        positives = [d.disease for d in details if d.is_positive]

        # Determine overall confidence
        if len(borderlines) >= 2 or any(d.confidence_level == ConfidenceLevel.BORDERLINE for d in details if d.disease in self.HIGH_ACUITY_DISEASES):
            overall_conf = ConfidenceLevel.BORDERLINE
        elif any(d.confidence_level in (ConfidenceLevel.LOW, ConfidenceLevel.BORDERLINE) for d in details):
            overall_conf = ConfidenceLevel.LOW
        elif mean_margin < self.MODERATE_MARGIN:
            overall_conf = ConfidenceLevel.MODERATE
        else:
            overall_conf = ConfidenceLevel.HIGH

        # Determine clinical recommendation
        # 1. High-acuity check:
        critical_positives = [d for d in positives if d in self.HIGH_ACUITY_DISEASES]
        if critical_positives:
            rec = ClinicalRecommendation.HIGH_ACUITY_ALERT
            reason = f"Critical finding(s) detected: {', '.join(critical_positives)}. Immediate radiologist review advised."
        elif len(borderlines) > 0 or has_quality_warnings:
            rec = ClinicalRecommendation.EXPERT_REVIEW_RECOMMENDED
            reasons = []
            if borderlines:
                reasons.append(f"Borderline findings for: {', '.join(borderlines)}")
            if has_quality_warnings:
                reasons.append("Image quality warnings flagged")
            reason = "; ".join(reasons) + ". Overread by qualified radiologist recommended."
        else:
            rec = ClinicalRecommendation.STANDARD_READING
            if positives:
                reason = f"Confirmed non-acute findings: {', '.join(positives)} with high margin confidence."
            else:
                reason = "No target pathologies exceeded decision threshold. Standard workflow reading."

        return UncertaintySummary(
            overall_confidence=overall_conf,
            mean_margin=mean_margin,
            borderline_diseases=borderlines,
            positive_findings=positives,
            recommendation=rec,
            recommendation_reason=reason
        )


uncertainty_analyzer = UncertaintyAnalyzer()
