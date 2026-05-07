"""
Gen AI Clinical Interpretation Assistant
=========================================================================
Provides AI-powered clinical interpretation of patient risk predictions
using Google's Gemini API. Falls back to template-based summaries when
no API key is configured.

Usage:
    from genai_assistant import ClinicalAssistant
    assistant = ClinicalAssistant()  # reads GEMINI_API_KEY from env
    result = assistant.interpret_predictions(patient_data, predictions)

Environment:
    GEMINI_API_KEY  — Google AI Studio API key (free tier: 60 req/min)
"""

import os
import json
import time
import hashlib
import traceback
from typing import Dict, Optional, List
from datetime import datetime


# ── Lazy import to avoid crash when google-generativeai is not installed ──────
_genai = None
_genai_available = None


def _ensure_genai():
    """Lazy-import google.generativeai; returns True if available."""
    global _genai, _genai_available
    if _genai_available is not None:
        return _genai_available
    try:
        import google.generativeai as genai
        _genai = genai
        _genai_available = True
    except ImportError:
        _genai_available = False
        print("[GENAI] google-generativeai not installed. AI features disabled.")
    return _genai_available


# ── Clinical System Prompt ────────────────────────────────────────────────────

CLINICAL_SYSTEM_PROMPT = """You are a clinical decision support AI assistant integrated into the Smart ICU Assistant system. Your role is to provide clear, evidence-based clinical interpretations of patient risk predictions.

IMPORTANT GUIDELINES:
1. You are a DECISION SUPPORT tool, NOT a replacement for clinical judgment.
2. Always include a disclaimer that final decisions must be made by qualified healthcare professionals.
3. Use evidence-based medical reasoning and cite clinical criteria where applicable (e.g., KDIGO for AKI, Sepsis-3, SIRS criteria).
4. Be concise but thorough. Use bullet points for clarity.
5. Tailor recommendations to the patient's ward/care unit type.
6. Flag urgent findings prominently.
7. Use standard medical terminology but keep it accessible.

WARD CONTEXT AWARENESS:
- Normal Ward patients: Focus on early warning signs, NEWS/MEWS scoring, ICU transfer criteria
- MICU patients: Focus on sepsis progression, respiratory failure, organ support
- SICU patients: Focus on surgical complications, bleeding, wound healing
- CCU patients: Focus on cardiac events, arrhythmia, hemodynamic stability
- CSRU patients: Focus on post-cardiac surgery recovery, AKI, ventilator weaning
- TSICU patients: Focus on trauma severity, hemorrhage control, secondary injuries

FORMAT your response as markdown with these sections:
##  Clinical Summary
Brief overview of the patient's risk profile.

##  Key Findings
Bullet points of the most important findings, ranked by urgency.

##  Suggested Interventions
Ward-appropriate interventions to consider.

##  Monitoring Recommendations
What to watch closely in the next 6-24 hours.

##  Escalation Criteria
When to escalate care or consider ICU transfer (for ward patients).
"""

HANDOFF_SYSTEM_PROMPT = """You are generating a clinical shift handoff summary (SBAR format) for an ICU patient. Be concise, factual, and highlight actionable items.

Format using SBAR:
## S — Situation
Current patient status and reason for ICU admission.

## B — Background  
Relevant medical history and recent trends.

## A — Assessment
Current risk levels and clinical concerns.

## R — Recommendation
Specific actions for the incoming team.
"""


class ClinicalAssistant:
    """AI-powered clinical interpretation using Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = None
        self._cache: Dict[str, dict] = {}
        self._cache_ttl = 300  # 5 min cache
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if Gen AI is configured and available."""
        return bool(self.api_key) and _ensure_genai()

    def _initialize(self):
        """Lazy initialization of the Gemini model."""
        if self._initialized:
            return
        if not self.is_available:
            self._initialized = True
            return

        try:
            _genai.configure(api_key=self.api_key)
            self.model = _genai.GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=CLINICAL_SYSTEM_PROMPT,
            )
            self._initialized = True
            print("[GENAI] Gemini model initialized successfully.")
        except Exception as e:
            print(f"[GENAI] Failed to initialize Gemini: {e}")
            self.model = None
            self._initialized = True

    def _cache_key(self, data: dict) -> str:
        """Generate a cache key from input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[dict]:
        """Get cached response if still valid."""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < self._cache_ttl:
                return entry["data"]
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: dict):
        """Cache a response."""
        self._cache[key] = {"data": data, "ts": time.time()}
        # Evict old entries if cache grows too large
        if len(self._cache) > 100:
            oldest = min(self._cache, key=lambda k: self._cache[k]["ts"])
            del self._cache[oldest]

    # ── Public API ────────────────────────────────────────────────────────────

    def interpret_predictions(
        self,
        patient_data: dict,
        predictions: dict,
        ward_type: str = "ICU",
    ) -> dict:
        """
        Generate a clinical interpretation of risk predictions.

        Returns:
            {
                "interpretation": str (markdown),
                "source": "gemini" | "template",
                "generated_at": str,
                "disclaimer": str,
            }
        """
        cache_input = {
            "patient": patient_data,
            "predictions": predictions,
            "ward_type": ward_type,
            "action": "interpret",
        }
        cache_key = self._cache_key(cache_input)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._initialize()

        if self.model:
            result = self._gemini_interpret(patient_data, predictions, ward_type)
        else:
            result = self._template_interpret(patient_data, predictions, ward_type)

        self._set_cache(cache_key, result)
        return result

    def generate_handoff(
        self,
        patient_data: dict,
        predictions: dict,
        vitals_summary: Optional[dict] = None,
        ward_type: str = "ICU",
    ) -> dict:
        """Generate a shift handoff summary in SBAR format."""
        cache_input = {
            "patient": patient_data,
            "predictions": predictions,
            "ward_type": ward_type,
            "action": "handoff",
        }
        cache_key = self._cache_key(cache_input)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._initialize()

        if self.model:
            result = self._gemini_handoff(patient_data, predictions, vitals_summary, ward_type)
        else:
            result = self._template_handoff(patient_data, predictions, ward_type)

        self._set_cache(cache_key, result)
        return result

    # ── Gemini API calls ──────────────────────────────────────────────────────

    def _gemini_interpret(self, patient_data: dict, predictions: dict, ward_type: str) -> dict:
        """Call Gemini API for clinical interpretation."""
        try:
            prompt = self._build_interpret_prompt(patient_data, predictions, ward_type)
            response = self.model.generate_content(prompt)
            text = response.text if response and response.text else ""

            if not text:
                return self._template_interpret(patient_data, predictions, ward_type)

            return {
                "interpretation": text,
                "source": "gemini",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "disclaimer": (
                    "⚕️ This AI-generated interpretation is for decision support only. "
                    "All clinical decisions must be made by qualified healthcare professionals. "
                    "This system does not replace clinical judgment."
                ),
            }
        except Exception as e:
            print(f"[GENAI] Gemini interpretation failed: {e}")
            traceback.print_exc()
            return self._template_interpret(patient_data, predictions, ward_type)

    def _gemini_handoff(
        self, patient_data: dict, predictions: dict,
        vitals_summary: Optional[dict], ward_type: str,
    ) -> dict:
        """Call Gemini API for handoff summary."""
        try:
            prompt = self._build_handoff_prompt(patient_data, predictions, vitals_summary, ward_type)

            # Use handoff-specific system prompt
            model = _genai.GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=HANDOFF_SYSTEM_PROMPT,
            )
            response = model.generate_content(prompt)
            text = response.text if response and response.text else ""

            if not text:
                return self._template_handoff(patient_data, predictions, ward_type)

            return {
                "handoff": text,
                "source": "gemini",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "format": "SBAR",
            }
        except Exception as e:
            print(f"[GENAI] Gemini handoff failed: {e}")
            traceback.print_exc()
            return self._template_handoff(patient_data, predictions, ward_type)

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_interpret_prompt(self, patient: dict, predictions: dict, ward_type: str) -> str:
        """Build the interpretation prompt with patient context."""

        # Extract key scores
        scores = predictions.get("scores", predictions)
        composite = predictions.get("composite_score", 0)
        risk_level = predictions.get("risk_level", "UNKNOWN")

        # Build patient summary
        lines = [
            f"**Ward/Unit Type:** {ward_type}",
            f"**Composite Risk Score:** {composite:.4f} ({risk_level})",
            "",
            "### Patient Demographics",
        ]

        # Demographics
        demo = patient.get("demographics", patient)
        if "age" in demo:
            lines.append(f"- Age: {demo['age']}")
        if "gender" in demo:
            lines.append(f"- Gender: {demo['gender']}")

        # Vitals
        vitals = patient.get("vitals", patient)
        vital_keys = ["heartrate", "sysbp", "diasbp", "meanbp", "resprate", "tempc", "spo2", "glucose"]
        vital_lines = []
        for k in vital_keys:
            if k in vitals and vitals[k] is not None:
                vital_lines.append(f"- {k}: {vitals[k]}")
        if vital_lines:
            lines.append("")
            lines.append("### Current Vitals")
            lines.extend(vital_lines)

        # Labs
        labs = patient.get("labs", patient)
        lab_keys = ["creatinine", "lactate", "wbc", "hemoglobin", "platelets", "bicarbonate"]
        lab_lines = []
        for k in lab_keys:
            if k in labs and labs[k] is not None:
                lab_lines.append(f"- {k}: {labs[k]}")
        if lab_lines:
            lines.append("")
            lines.append("### Laboratory Values")
            lines.extend(lab_lines)

        # Prediction scores
        lines.append("")
        lines.append("### Risk Prediction Scores")
        groups = predictions.get("groups", {})
        for group_name, labels in groups.items():
            lines.append(f"**{group_name}:**")
            for lbl, score in labels.items():
                pct = score * 100
                flag = " ⚠️" if score > 0.6 else " ⚡" if score > 0.3 else ""
                lines.append(f"  - {lbl}: {pct:.1f}%{flag}")

        # Clinical scores if available
        clinical = predictions.get("clinical_scores", {})
        if clinical:
            lines.append("")
            lines.append("### Clinical Markers")
            for k, v in clinical.items():
                lines.append(f"- {k}: {v}")

        # Alerts
        alerts = predictions.get("alerts", [])
        if alerts:
            lines.append("")
            lines.append("### Active Alerts")
            for a in alerts:
                lines.append(f"- [{a.get('type', 'info').upper()}] {a.get('category', '')}: {a.get('message', '')}")

        prompt = "\n".join(lines)
        prompt += "\n\nPlease provide a clinical interpretation of this patient's risk profile, tailored to their ward type."
        return prompt

    def _build_handoff_prompt(
        self, patient: dict, predictions: dict,
        vitals_summary: Optional[dict], ward_type: str,
    ) -> str:
        """Build the SBAR handoff prompt."""
        base = self._build_interpret_prompt(patient, predictions, ward_type)
        extra = "\n\nGenerate a concise SBAR shift handoff summary for this patient."
        if vitals_summary:
            extra += f"\n\nVitals trend summary: {json.dumps(vitals_summary, default=str)}"
        return base + extra

    # ── Template-based fallbacks ──────────────────────────────────────────────

    def _template_interpret(self, patient: dict, predictions: dict, ward_type: str) -> dict:
        """Generate a template-based interpretation when Gen AI is unavailable."""
        scores = predictions.get("scores", predictions)
        composite = predictions.get("composite_score", 0)
        risk_level = predictions.get("risk_level", "UNKNOWN")
        groups = predictions.get("groups", {})
        clinical = predictions.get("clinical_scores", {})

        sections = []

        # Clinical Summary
        sections.append("## 🔍 Clinical Summary")
        sections.append(
            f"This {ward_type} patient has a **composite risk score of {composite:.4f}** "
            f"({risk_level} risk). "
        )
        if risk_level == "HIGH":
            sections.append("⚠️ **Immediate clinical review recommended.**")
        elif risk_level == "MEDIUM":
            sections.append("Close monitoring is advised with reassessment in 2-4 hours.")
        else:
            sections.append("Current risk levels are within acceptable ranges. Continue routine monitoring.")

        # Key Findings
        sections.append("\n## ⚠️ Key Findings")
        high_risks = []
        moderate_risks = []
        for group_name, labels in groups.items():
            for lbl, score in labels.items():
                if score > 0.6:
                    high_risks.append((lbl, score, group_name))
                elif score > 0.3:
                    moderate_risks.append((lbl, score, group_name))

        if high_risks:
            for lbl, score, grp in sorted(high_risks, key=lambda x: -x[1]):
                sections.append(f"- 🔴 **{grp} — {lbl}**: {score*100:.1f}% (HIGH)")
        if moderate_risks:
            for lbl, score, grp in sorted(moderate_risks, key=lambda x: -x[1]):
                sections.append(f"- 🟡 **{grp} — {lbl}**: {score*100:.1f}% (MODERATE)")
        if not high_risks and not moderate_risks:
            sections.append("- 🟢 No elevated risk predictions detected.")

        # SIRS
        sirs = clinical.get("sirs", 0)
        if sirs >= 2:
            sections.append(f"- 🔴 SIRS Score: **{sirs}/4** — {'meets sepsis screening criteria' if sirs >= 3 else 'monitor for progression'}")

        shock_idx = clinical.get("shock_index", 0)
        if shock_idx and shock_idx > 1.0:
            sections.append(f"- 🔴 Shock Index: **{shock_idx}** (>1.0 — hemodynamic instability)")

        # Interventions
        sections.append("\n## 💊 Suggested Interventions")
        if any(s > 0.6 for _, s, _ in high_risks):
            sections.append("- Consider urgent clinical review and bedside assessment")
        if scores.get("sepsis_24h", 0) > 0.5:
            sections.append("- Initiate sepsis bundle (blood cultures, lactate, IV fluids) per Sepsis-3 guidelines")
        if scores.get("vasopressor_6h", 0) > 0.5:
            sections.append("- Assess fluid responsiveness; prepare vasopressor support if MAP remains <65 mmHg")
        if scores.get("ventilation_6h", 0) > 0.5:
            sections.append("- Monitor respiratory status closely; consider NIV or intubation if SpO₂ <90%")
        if scores.get("aki_stage1_24h", 0) > 0.5:
            sections.append("- Optimize renal perfusion; avoid nephrotoxins; monitor urine output per KDIGO criteria")
        if not high_risks:
            sections.append("- Continue current management plan")
            sections.append("- Routine vital sign monitoring per unit protocol")

        # Monitoring
        sections.append("\n## 📊 Monitoring Recommendations")
        if risk_level == "HIGH":
            sections.append("- Continuous cardiac monitoring")
            sections.append("- Vital signs every 15-30 minutes")
            sections.append("- Hourly urine output monitoring")
            sections.append("- Repeat labs in 4-6 hours")
        elif risk_level == "MEDIUM":
            sections.append("- Vital signs every 1-2 hours")
            sections.append("- Reassess clinical status in 2-4 hours")
            sections.append("- Repeat labs in 6-8 hours")
        else:
            sections.append("- Routine vital sign monitoring per unit protocol")
            sections.append("- Standard lab schedule")

        # Escalation (especially for ward patients)
        if ward_type in ("WARD", "Normal Ward"):
            sections.append("\n## ⚡ ICU Transfer Criteria")
            sections.append("Consider ICU transfer if:")
            sections.append("- Composite risk score exceeds 0.60")
            sections.append("- MAP < 65 mmHg despite fluid resuscitation")
            sections.append("- SpO₂ < 90% on supplemental oxygen")
            sections.append("- New onset organ dysfunction (rising creatinine, altered mental status)")
            sections.append("- SIRS ≥ 3 with suspected infection")
            if composite > 0.4:
                sections.append(f"\n> ⚠️ **Current composite score ({composite:.4f}) suggests this patient may benefit from ICU-level monitoring.**")
        else:
            sections.append("\n## ⚡ Escalation Criteria")
            sections.append("- Notify attending if risk scores increase by >20% on reassessment")
            sections.append("- Consider specialty consultation for persistent high-risk predictions")

        interpretation = "\n".join(sections)

        return {
            "interpretation": interpretation,
            "source": "template",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "disclaimer": (
                "⚕️ This is an automated clinical summary based on prediction scores. "
                "It is not AI-generated. For AI-powered interpretations, configure a "
                "GEMINI_API_KEY environment variable. All clinical decisions must be made "
                "by qualified healthcare professionals."
            ),
        }

    def _template_handoff(self, patient: dict, predictions: dict, ward_type: str) -> dict:
        """Generate a template-based SBAR handoff."""
        scores = predictions.get("scores", predictions)
        composite = predictions.get("composite_score", 0)
        risk_level = predictions.get("risk_level", "UNKNOWN")
        demo = patient.get("demographics", patient)
        clinical = predictions.get("clinical_scores", {})

        age = demo.get("age", "?")
        gender = demo.get("gender", "?")
        diagnosis = patient.get("history", patient).get("diagnosis", "") or patient.get("diagnosis", "")

        lines = []

        # Situation
        lines.append("## S — Situation")
        lines.append(
            f"{age}-year-old {gender} patient in **{ward_type}** "
            f"with composite risk score **{composite:.4f}** ({risk_level})."
        )
        if diagnosis:
            lines.append(f"Primary diagnosis: {diagnosis}")

        # Background
        lines.append("\n## B — Background")
        sirs = clinical.get("sirs", 0)
        si = clinical.get("shock_index", 0)
        lines.append(f"- SIRS Score: {sirs}/4")
        if si:
            lines.append(f"- Shock Index: {si}")

        high_items = [
            (lbl, s) for lbl, s in scores.items()
            if s > 0.5 and not lbl.startswith("los_")
        ]
        if high_items:
            lines.append("- Elevated risk predictions:")
            for lbl, s in sorted(high_items, key=lambda x: -x[1]):
                lines.append(f"  - {lbl}: {s*100:.1f}%")

        # Assessment
        lines.append("\n## A — Assessment")
        if risk_level == "HIGH":
            lines.append("Patient is in a **high-risk** state requiring close monitoring.")
        elif risk_level == "MEDIUM":
            lines.append("Patient has **moderate risk** — monitor for changes.")
        else:
            lines.append("Patient is currently **stable** with low risk scores.")

        # Recommendation
        lines.append("\n## R — Recommendation")
        if risk_level == "HIGH":
            lines.append("- Continue current interventions; escalate if deterioration")
            lines.append("- Vital signs every 15-30 minutes")
            lines.append("- Repeat labs in 4 hours")
        elif risk_level == "MEDIUM":
            lines.append("- Vital signs every 1-2 hours")
            lines.append("- Reassess in 2-4 hours")
        else:
            lines.append("- Routine monitoring per unit protocol")

        return {
            "handoff": "\n".join(lines),
            "source": "template",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "format": "SBAR",
        }
