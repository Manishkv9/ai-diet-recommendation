import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Human-readable labels for dataset abbreviations
_FIELD_LABELS = {
    "FAVC": "high-calorie food consumption",
    "FCVC": "vegetable intake frequency",
    "NCP": "meals per day",
    "CAEC": "snacking frequency",
    "CH2O": "daily water intake",
    "SCC": "calorie tracking",
    "FAF": "physical activity frequency",
    "TUE": "screen time",
    "CALC": "alcohol consumption",
    "MTRANS": "transportation method",
}

# Readable mappings for numeric coded values
_WATER_LEVELS = {"1.0": "less than 1 liter", "2.0": "1–2 liters", "3.0": "more than 2 liters"}
_ACTIVITY_LEVELS = {"0.0": "no physical activity", "1.0": "1–2 days/week", "2.0": "2–4 days/week", "3.0": "4–5 days/week"}
_VEGGIE_LEVELS = {"1.0": "rarely", "2.0": "sometimes", "3.0": "always"}
_MEAL_COUNTS = {"1.0": "1–2 meals", "2.0": "3 meals", "3.0": "more than 3 meals", "4.0": "4+ meals"}


def _build_user_profile_summary(user_data, bmi, bmi_category):
    """Builds a concise text summary of the user's actual inputs for the prompt."""
    height = user_data.get("Height", "unknown")
    weight = user_data.get("Weight", "unknown")
    age = user_data.get("Age", "unknown")
    gender = user_data.get("Gender", "unknown")

    water_raw = str(user_data.get("CH2O", ""))
    water = _WATER_LEVELS.get(water_raw, water_raw)

    activity_raw = str(user_data.get("FAF", ""))
    activity = _ACTIVITY_LEVELS.get(activity_raw, activity_raw)

    veggie_raw = str(user_data.get("FCVC", ""))
    veggies = _VEGGIE_LEVELS.get(veggie_raw, veggie_raw)

    meals_raw = str(user_data.get("NCP", ""))
    meals = _MEAL_COUNTS.get(meals_raw, meals_raw)

    family_history = user_data.get("family_history_with_overweight", "unknown")

    lines = [
        f"- Gender: {gender}, Age: {age}",
        f"- Height: {height} m, Weight: {weight} kg",
        f"- BMI: {bmi} ({bmi_category})",
        f"- Daily water intake: {water}",
        f"- Physical activity: {activity}",
        f"- Vegetable intake: {veggies}",
        f"- Meals per day: {meals}",
        f"- Family history of overweight: {family_history}",
    ]
    return "\n".join(lines)


def generate_diet_explanation(obesity_prediction, diet_goal, meal_plan,
                               user_data=None, bmi=0, bmi_category="Unknown"):
    """
    Generates a professional, concise dietitian-style explanation for the diet plan
    using Google's Gemini 2.5 Flash model.
    
    Args:
        obesity_prediction (str): The predicted obesity class.
        diet_goal (str): The target diet goal.
        meal_plan (dict): The generated breakfast, lunch, and dinner recommendations.
        user_data (dict, optional): The user's raw form inputs for personalized context.
        bmi (float): The user's calculated BMI.
        bmi_category (str): The BMI classification string.
        
    Returns:
        str: A clean, professional paragraph suitable for UI display.
    """
    try:
        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_key_here":
            return "AI Explanation unavailable: Please set a valid GEMINI_API_KEY in your .env file."
            
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize Gemini 2.5 Flash model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build the user profile summary
        profile_summary = ""
        if user_data:
            profile_summary = _build_user_profile_summary(user_data, bmi, bmi_category)
        
        # Construct the professional dietitian prompt
        prompt = f"""You are a clinical dietitian writing a formal medical analysis for a patient's chart and for the patient to read.

The patient's profile:
{profile_summary}

The ML model predicted their weight category as: '{obesity_prediction}'.
Based on this, the recommended diet goal is: '{diet_goal}'.

Write a concise clinical analysis paragraph (120–180 words). Follow these rules STRICTLY:

1. DO NOT use ANY emotional, empathetic, or motivational phrases. Completely ban phrases like: "It's wonderful", "Congratulations", "Don't worry", "It's completely normal", "Great news", "I understand", or "We are here to help".
2. Start DIRECTLY with the health analysis. Your first sentence MUST explicitly reference the patient's exact BMI ({bmi}, {bmi_category}) and clearly state how their specific lifestyle indicators contributed to the '{obesity_prediction}' classification.
3. Reference SPECIFIC user inputs: their weight, height, water intake, activity level, vegetable consumption, and meal frequency. Use the actual values provided above.
4. Explain clearly WHY the model predicted the '{obesity_prediction}' category based on the combination of their inputs.
5. Explain clearly WHY the '{diet_goal}' goal was selected as the appropriate therapeutic response.
6. End with exactly 2–3 specific, actionable recommendations as a brief numbered or bulleted list. Examples: "Increase daily water intake to 2–3 liters", "Include vegetables in at least two meals", "Aim for 30 minutes of physical activity daily".
7. Sound like a clinical healthcare provider communicating medical facts. Be direct, objective, and evidence-based.
8. Output ONLY the clean text. No markdown formatting, no bolding, no headings, no greetings."""
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Return the generated text, stripped of leading/trailing whitespace
        return response.text.strip()
        
    except Exception as e:
        # Handle API errors, quota limits, or network issues gracefully
        print(f"Gemini API Error: {str(e)}")
        return (
            f"Based on your BMI of {bmi} ({bmi_category}), the model classified your weight status as "
            f"{obesity_prediction.replace('_', ' ')}. The recommended dietary approach is {diet_goal}. "
            f"Focus on balanced meals with adequate protein and vegetables, stay hydrated with at least "
            f"2 liters of water daily, and aim for regular physical activity."
        )
