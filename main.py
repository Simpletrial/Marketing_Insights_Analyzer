import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION")
)

# Input Feedbacks
feedbacks = [
    "Delivery was slow but customer support was helpful",
    "The app crashes frequently after the update",
    "Customer service resolved my issue quickly",
    "Great features but onboarding is confusing",
    "I absolutely love the app – smooth performance, great features, and excellent support"
]

print("\nInput Feedbacks:")
for i, fb in enumerate(feedbacks, start=1):
    print(f"{i}. {fb}")

# ---------------- RULE-BASED ANALYZER ----------------
def rule_based_analyzer(text):
    text_lower = text.lower()
    sentiment = "Neutral"
    themes = []
    complaints = []
    improvements = []
    summary_points = []

    if "slow" in text_lower or "delay" in text_lower:
        complaints.append("Delivery delay")
        themes.append("Delivery")
        improvements.append("Investigate delivery processes")
        summary_points.append("Delivery delays detected")

    if "crash" in text_lower:
        complaints.append("App stability issues")
        themes.append("Technical")
        improvements.append("Fix app crashes")
        summary_points.append("Application crashes reported")

    if "onboarding" in text_lower or "confusing" in text_lower:
        complaints.append("Onboarding process is confusing")
        themes.append("Onboarding")
        improvements.append("Simplify onboarding experience")
        summary_points.append("Onboarding issues detected")

    if "helpful" in text_lower or "resolved" in text_lower or "great" in text_lower:
        themes.append("Support")
        improvements.append("Acknowledge positive feedback")
        summary_points.append("Positive customer experience noted")

    if complaints and summary_points:
        sentiment = "Mixed"
    elif complaints:
        sentiment = "Negative"
    elif themes:
        sentiment = "Positive"

    return {
        "Sentiment": sentiment,
        "Summary": "; ".join(summary_points),
        "Themes": list(set(themes)),
        "Complaints": list(set(complaints)),
        "Improvement Suggestions": list(set(improvements))
    }

# ---------------- AI-BASED ANALYZER ----------------
def ai_analyzer(text):
    prompt = f"""
You are an expert customer feedback analyst.

Analyze the feedback and return a VALID JSON object with EXACTLY these keys:
- sentiment
- summary_of_key_points
- themes
- complaints
- improvement_suggestions

Rules:
- Extract complaints ONLY from the feedback.
- If a complaint exists, suggest at least one improvement.
- Do NOT return None.
- Use empty lists only if nothing applies.

Customer feedback:
{text}
"""

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You generate structured customer insights."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content

    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:-1])

    data = json.loads(content)

    return {
        "Sentiment": data["sentiment"].capitalize(),
        "Summary": data["summary_of_key_points"],
        "Themes": data["themes"],
        "Complaints": data["complaints"],
        "Improvement Suggestions": data["improvement_suggestions"]
    }

# ---------------- HELPER FUNCTION ----------------
def format_list(values):
    if not values:
        return "None"
    return ", ".join(f'"{v}"' for v in values)

# ---------------- MAIN PROCESS ----------------
for feedback in feedbacks:
    rule = rule_based_analyzer(feedback)
    ai = ai_analyzer(feedback)

    final_sentiment = ai["Sentiment"]
    final_summary = ai["Summary"]
    final_themes = list(set(rule["Themes"] + ai["Themes"]))
    final_complaints = list(set(rule["Complaints"] + ai["Complaints"]))
    final_improvements = list(set(rule["Improvement Suggestions"] + ai["Improvement Suggestions"]))

    print("{")
    print(f'  Feedback: "{feedback}",\n')

    print(f'  Rule-Based Sentiment: "{rule["Sentiment"]}",')
    print(f'  Rule-Based Summary: "{rule["Summary"]}",')
    print(f'  Rule-Based Themes: {format_list(rule["Themes"])},')
    print(f'  Rule-Based Complaints: {format_list(rule["Complaints"])},')
    print(f'  Rule-Based Improvement Suggestions: {format_list(rule["Improvement Suggestions"])},\n')

    print(f'  AI-Based Sentiment: "{ai["Sentiment"]}",')
    print(f'  AI-Based Summary: "{ai["Summary"]}",')
    print(f'  AI-Based Themes: {format_list(ai["Themes"])},')
    print(f'  AI-Based Complaints: {format_list(ai["Complaints"])},')
    print(f'  AI-Based Improvement Suggestions: {format_list(ai["Improvement Suggestions"])},\n')

    print(f'  ➡️ Final Decision')
    print(f'  Sentiment: "{final_sentiment}"')
    print(f'  Summary: "{final_summary}"')
    print(f'  Themes: {format_list(final_themes)}')
    print(f'  Complaints: {format_list(final_complaints)}')
    print(f'  Improvement Suggestions: {format_list(final_improvements)}')
    print("},")
