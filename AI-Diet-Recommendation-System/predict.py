import os
import pickle
import pandas as pd
import json

# Import backend modules
from backend.diet_mapping import get_diet_goal
from backend.meal_recommender import recommend_meals
from backend.gemini_service import generate_diet_explanation

def predict_diet(user_data, bmi=0, bmi_category="Unknown"):
    """
    Predicts the obesity category for a user, maps it to a diet goal,
    and returns a structured meal plan.
    
    Args:
        user_data (dict): A dictionary containing the user's attributes.
                          Keys must match the features in the obesity dataset.
        bmi (float): Pre-calculated BMI from the frontend.
        bmi_category (str): BMI classification (Underweight/Normal/Overweight/Obese).
                          
    Returns:
        dict: Containing the obesity prediction, diet goal, and meal plan.
    """
    try:
        # 1. Load the trained model, target encoder, and feature names
        models_dir = "models"
        model_path = os.path.join(models_dir, "best_model.pkl")
        encoder_path = os.path.join(models_dir, "target_encoder.pkl")
        features_path = os.path.join(models_dir, "feature_names.pkl")
        
        # Check if model files exist
        for path in [model_path, encoder_path, features_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required model file missing: {path}. Please run train_model.py first.")
                
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        with open(encoder_path, "rb") as f:
            target_encoder = pickle.load(f)
            
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)
            
        # 2. Preprocess user data
        # Convert single user dictionary to a pandas DataFrame
        df_user = pd.DataFrame([user_data])
        
        # Apply feature engineering
        from train_model import feature_engineering
        df_user = feature_engineering(df_user)
        
        # Categorical columns used during training
        categorical_cols = [
            'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
            'SMOKE', 'SCC', 'CALC', 'MTRANS', 'Age_Group', 'Activity_Level', 'Water_Intake_Category'
        ]
        
        # Only encode categorical columns that are present in the user_data
        cols_to_encode = [c for c in categorical_cols if c in df_user.columns]
        
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df_user, columns=cols_to_encode, drop_first=True)
        
        # Reindex to ensure the input data has exactly the same columns as the training data.
        # This adds missing dummy columns with 0 and removes extra columns.
        df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
        
        # Ensure data types are numeric (important for models like XGBoost)
        df_encoded = df_encoded.astype(float)
        
        # 3. Predict obesity category
        prediction_idx = model.predict(df_encoded)[0]
        
        # Map the numeric prediction back to the original string label
        obesity_category = target_encoder.inverse_transform([prediction_idx])[0]
        
        # 4. Convert obesity category to diet goal
        diet_goal = get_diet_goal(obesity_category)
        
        # 5. Generate meal recommendations
        meal_plan = recommend_meals(diet_goal, obesity_prediction=obesity_category)
        
        # 6. Generate AI Explanation using Gemini
        # Pass user_data and BMI context so Gemini can reference actual inputs
        ai_explanation = generate_diet_explanation(
            obesity_prediction=obesity_category,
            diet_goal=diet_goal,
            meal_plan=meal_plan,
            user_data=user_data,
            bmi=bmi,
            bmi_category=bmi_category
        )
        
        # Return the structured response
        return {
            "obesity_prediction": obesity_category,
            "diet_goal": diet_goal,
            "meal_plan": meal_plan,
            "ai_explanation": ai_explanation
        }
        
    except Exception as e:
        # Handle exceptions gracefully
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":
    # Sample user example
    sample_user = {
        "Gender": "Male",
        "Age": 25,
        "Height": 1.75,
        "Weight": 115.0, # Indicates potential obesity
        "family_history_with_overweight": "yes",
        "FAVC": "yes",
        "FCVC": 2.0,
        "NCP": 3.0,
        "CAEC": "Sometimes",
        "SMOKE": "no",
        "CH2O": 2.0,
        "SCC": "no",
        "FAF": 1.0,
        "TUE": 1.0,
        "CALC": "no",
        "MTRANS": "Public_Transportation"
    }
    
    # Calculate BMI for the sample user
    sample_bmi = sample_user["Weight"] / (sample_user["Height"] ** 2)
    sample_bmi_cat = "Obese" if sample_bmi >= 30 else "Overweight" if sample_bmi >= 25 else "Normal" if sample_bmi >= 18.5 else "Underweight"
    
    print("========================================")
    print("Testing Diet Recommendation System")
    print("========================================")
    print("Input User Data:")
    print(json.dumps(sample_user, indent=4))
    print(f"\nBMI: {sample_bmi:.1f} ({sample_bmi_cat})")
    print("\nProcessing...")
    
    # Run prediction
    result = predict_diet(sample_user, bmi=sample_bmi, bmi_category=sample_bmi_cat)
    
    print("\n========================================")
    print("Result:")
    print("========================================")
    print(json.dumps(result, indent=4))
