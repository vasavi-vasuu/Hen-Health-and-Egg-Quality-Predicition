import os
from typing import Dict
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

# ---------------------------------------------------------
# GEMINI API KEY
# ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------
# GEMINI MODEL
# ---------------------------------------------------------
MODEL_NAME = "gemini-3.6-flash"


def is_gemini_configured() -> bool:
    """Check whether a Gemini API key is available."""
    return bool(GEMINI_API_KEY)


def generate_ai_report(
    subject: str,
    prediction: str,
    confidence: float,
    reference_condition: str,
    reference_note: str,
) -> Dict[str, str]:

    # Check API key
    if not GEMINI_API_KEY:
        return {
            "ok": False,
            "error": (
                "Gemini API key is not configured. "
                "Please create a .env file and add GEMINI_API_KEY."
            ),
        }

    prompt = f"""
You are an educational poultry-care AI assistant.

Generate a simple and professional report based on this
machine-learning prediction.

Subject:
{subject}

Prediction:
{prediction}

Model confidence:
{confidence:.1f}%

Reference condition:
{reference_condition}

Reference note:
{reference_note}

Please provide:

1. Prediction Summary
2. What the result may indicate
3. Possible causes
4. Recommended care and observation
5. When veterinary consultation may be appropriate

Use simple language suitable for a college project demonstration.

Important:
This is an educational AI/ML project and not a veterinary diagnosis.
Do not claim laboratory confirmation.
Do not invent test results.
"""

    try:

        # Create Gemini client using the hidden environment key
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        report = response.text

        if not report:
            return {
                "ok": False,
                "error": "Gemini returned an empty response.",
            }

        return {
    "ok": True,
    "text": report,
    "model": MODEL_NAME,
}

    except Exception as exc:

        return {
            "ok": False,
            "error": f"Gemini report generation failed: {exc}",
        }