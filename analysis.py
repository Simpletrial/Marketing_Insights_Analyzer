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

# ---------------- READ FEEDBACKS FROM FILE ----------------
feedback_file = "feedbacks.txt"
with open(feedback_file, "r", encoding="utf-8") as f:
    feedbacks = [line.strip() for line in f if line.strip()]

print("\n📥 Feedbacks successfully imported from feedbacks.txt")
for i, fb in enumerate(feedbacks, 1):
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
        complaints.append("Delivery was delayed")
        themes.append("Delivery")
        improvements.append("Improve delivery timelines")
        summary_points.append("Delivery issues detected")

    if "crash" in text_lower:
        complaints.append("Application crashes reported")
        themes.append("App Stability")
        improvements.append("Fix crashes introduced in the update")
        summary_points.append("App crashes reported")

    if "onboarding" in text_lower or "confusing" in text_lower:
        complaints.append("Onboarding process is confusing")
        themes.append("Onboarding")
        improvements.append("Simplify onboarding experience")
        summary_points.append("Onboarding issues detected")

    if "helpful" in text_lower or "resolved" in text_lower or "great" in text_lower:
        themes.append("Customer Support")
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
        "Summary": "; ".join(summary_points) if summary_points else "None",
        "Themes": list(set(themes)) if themes else ["None"],
        "Complaints": list(set(complaints)) if complaints else ["None"],
        "Improvement Suggestions": list(set(improvements)) if improvements else ["None"]
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
        "Summary": data["summary_of_key_points"] if data["summary_of_key_points"] else "None",
        "Themes": data["themes"] if data["themes"] else ["None"],
        "Complaints": data["complaints"] if data["complaints"] else ["None"],
        "Improvement Suggestions": data["improvement_suggestions"] if data["improvement_suggestions"] else ["None"]
    }

# ---------------- HELPER FUNCTION ----------------
def format_list(values):
    if not values or values == ["None"]:
        return "None"
    return ", ".join(f'"{v}"' for v in values)

# ---------------- FINAL DECISION FUNCTION ----------------
def generate_final_decision(rule, ai):
    # 1. Sentiment
    if rule["Sentiment"] == ai["Sentiment"]:
        final_sentiment = rule["Sentiment"]
    else:
        final_sentiment = "Negative" if "Negative" in [rule["Sentiment"], ai["Sentiment"]] else "Mixed"

    # 2. Summary - prefer AI summary if available
    final_summary = ai["Summary"] if ai["Summary"] != "None" else rule["Summary"]

    # 3. Themes - keep meaningful common concepts
    final_themes = []
    for t in ai["Themes"]:
        if any(t.lower() in r.lower() or r.lower() in t.lower() for r in rule["Themes"]):
            final_themes.append(t)
    if not final_themes or final_themes == ["None"]:
        final_themes = ai["Themes"] if ai["Themes"] != ["None"] else rule["Themes"]

    # 4. Complaints - combine AI + rule
    final_complaints = []
    for c in ai["Complaints"]:
        if any(c.lower() in r.lower() or r.lower() in c.lower() for r in rule["Complaints"]):
            final_complaints.append(c)
    if not final_complaints or final_complaints == ["None"]:
        final_complaints = ai["Complaints"] if ai["Complaints"] != ["None"] else rule["Complaints"]

    # 5. Improvement Suggestions - combine AI + rule
    final_improvements = []
    for imp in ai["Improvement Suggestions"]:
        if any(imp.lower() in r.lower() or r.lower() in imp.lower() for r in rule["Improvement Suggestions"]):
            final_improvements.append(imp)
    if not final_improvements or final_improvements == ["None"]:
        final_improvements = ai["Improvement Suggestions"] if ai["Improvement Suggestions"] != ["None"] else rule["Improvement Suggestions"]

    return final_sentiment, final_summary, final_themes, final_complaints, final_improvements

# ---------------- MAIN PROCESS ----------------
for feedback in feedbacks:
    rule = rule_based_analyzer(feedback)
    ai = ai_analyzer(feedback)

    final_sentiment, final_summary, final_themes, final_complaints, final_improvements = generate_final_decision(rule, ai)

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

    print(f'  ➡️ Final Decision ')
    print(f'  Sentiment: "{final_sentiment}"')
    print(f'  Summary: "{final_summary}"')
    print(f'  Themes: {format_list(final_themes)}')
    print(f'  Complaints: {format_list(final_complaints)}')
    print(f'  Improvement Suggestions: {format_list(final_improvements)}')
    print("},")
