"""
diet_mapping.py
Maps obesity classes predicted by the machine learning model to actionable diet goals.
"""

# Dictionary mapping obesity classes to specific diet goals
DIET_GOAL_MAPPING = {
    "Insufficient_Weight": "Weight Gain",
    "Normal_Weight": "Maintenance",
    "Overweight_Level_I": "Weight Loss",
    "Overweight_Level_II": "Weight Loss",
    "Obesity_Type_I": "Aggressive Weight Loss",
    "Obesity_Type_II": "Aggressive Weight Loss",
    "Obesity_Type_III": "Medical Weight Loss"
}

def get_diet_goal(obesity_class):
    """
    Returns the mapped diet goal for a given obesity classification.
    
    Args:
        obesity_class (str): The obesity classification predicted by the model 
                             (e.g., 'Normal_Weight', 'Obesity_Type_I').
                             
    Returns:
        str: The recommended diet goal.
        
    Raises:
        ValueError: If the provided obesity_class is not in the mapping dictionary.
    """
    if obesity_class not in DIET_GOAL_MAPPING:
        raise ValueError(f"Unknown obesity class: {obesity_class}. Expected one of {list(DIET_GOAL_MAPPING.keys())}")
    
    return DIET_GOAL_MAPPING[obesity_class]
