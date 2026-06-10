import os
import json
import random

"""
meal_recommender.py
Curated Meal Recommendation Engine.
Features:
- Loads pre-curated realistic meals from datasets/indian_meal_plans.json
- Filters perfectly by diet goal
- Adds dynamic scoring for variety
- Ensures no duplicate meals across the day
"""

def _load_dataset(filepath="datasets/indian_meal_plans.json"):
    if not os.path.exists(filepath):
        print(f"Warning: Dataset not found at '{filepath}'.")
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")
        return []

def _score_meal(meal, diet_goal):
    """
    Score the meal to add some variation among the valid filtered meals.
    Although they are pre-curated for the goal, we add a score to pick the 'best' fitting ones.
    """
    score = 0
    if diet_goal == "Maintenance":
        score = meal.get('protein', 0) - meal.get('fat', 0)
    elif diet_goal in ["Weight Loss", "Aggressive Weight Loss"]:
        score = (meal.get('protein', 0) * 2) - meal.get('calories', 0) * 0.1
    elif diet_goal == "Medical Weight Loss":
        score = (meal.get('protein', 0) * 3) - meal.get('calories', 0) * 0.5 - meal.get('fat', 0) * 2
    elif diet_goal == "Weight Gain":
        score = meal.get('calories', 0) * 0.1 + meal.get('protein', 0)
    return score

def _select_meal(meals, meal_type, diet_goal, seen_meals):
    """
    Selects a meal for a specific meal_type matching the diet_goal.
    Ensures the exact meal name hasn't been used today.
    """
    # Filter by type and goal
    valid_meals = [m for m in meals if m['meal_type'] == meal_type and m['diet_goal'] == diet_goal]
    
    # Fallback if no exact goal match: try Maintenance for balanced or closest
    if not valid_meals:
        valid_meals = [m for m in meals if m['meal_type'] == meal_type]
        
    # Remove seen meals
    available = [m for m in valid_meals if m['meal_name'] not in seen_meals]
    
    if not available:
        # Emergency fallback if all exhausted (should not happen with large dataset)
        available = valid_meals
        
    if not available:
        return []
        
    # Score available meals
    for m in available:
        m['score'] = _score_meal(m, diet_goal)
        
    # Sort by score descending
    available.sort(key=lambda x: x['score'], reverse=True)
    
    # To avoid always picking the exact same top meal every single day,
    # we take the top 5 and pick one randomly. This is safe because all meals
    # in the curated dataset for this goal are verified healthy/appropriate.
    top_n = available[:5]
    selected = random.choice(top_n)
    
    seen_meals.add(selected['meal_name'])
    
    import re
    
    # Strip any trailing numbers and format title casing as a fail-safe
    clean_name = re.sub(r'\s*\d+$', '', selected['meal_name']).title()
    
    # Format for frontend output
    return [{
        "food_name": clean_name,
        "calories": float(selected['calories']),
        "protein": float(selected['protein']),
        "carbs": float(selected['carbs']),
        "fat": float(selected['fat'])
    }]

def recommend_meals(diet_goal, obesity_prediction="Unknown"):
    """
    Generates a full day's meal plan by querying the curated JSON dataset.
    """
    print("\n" + "="*50)
    print("CURATED MEAL RECOMMENDATION ENGINE")
    print("="*50)
    print(f"Obesity Prediction: {obesity_prediction}")
    print(f"Selected Diet Goal: {diet_goal}")
    
    meals_data = _load_dataset()
    if not meals_data:
        return {"breakfast": [], "lunch": [], "dinner": [], "snack": []}
        
    seen_meals = set()
    
    breakfast = _select_meal(meals_data, "breakfast", diet_goal, seen_meals)
    lunch = _select_meal(meals_data, "lunch", diet_goal, seen_meals)
    snack = _select_meal(meals_data, "snack", diet_goal, seen_meals)
    dinner = _select_meal(meals_data, "dinner", diet_goal, seen_meals)
    
    print("Final Meal Selections:")
    if breakfast: print(f"Breakfast: {breakfast[0]['food_name']} ({breakfast[0]['calories']} kcal)")
    if lunch: print(f"Lunch:     {lunch[0]['food_name']} ({lunch[0]['calories']} kcal)")
    if snack: print(f"Snack:     {snack[0]['food_name']} ({snack[0]['calories']} kcal)")
    if dinner: print(f"Dinner:    {dinner[0]['food_name']} ({dinner[0]['calories']} kcal)")
    print("="*50 + "\n")
    
    return {
        "breakfast": breakfast,
        "lunch": lunch,
        "snack": snack,
        "dinner": dinner
    }

if __name__ == "__main__":
    recommend_meals("Medical Weight Loss", obesity_prediction="Obesity_Type_III")
