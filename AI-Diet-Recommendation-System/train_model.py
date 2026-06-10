import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def feature_engineering(df):
    """
    Applies feature engineering to the dataset.
    This function should be used in both training and prediction pipelines.
    """
    df = df.copy()
    
    # Calculate BMI
    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        
    # Age Groups
    if 'Age' in df.columns:
        bins = [0, 20, 30, 40, 100]
        labels = ['Under_20', '20-30', '30-40', '40_Plus']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        # Convert to string to treat as categorical
        df['Age_Group'] = df['Age_Group'].astype(str)
        
    # Activity Level Categories (FAF)
    if 'FAF' in df.columns:
        bins = [-1, 1, 2, 4]
        labels = ['Low', 'Moderate', 'High']
        df['Activity_Level'] = pd.cut(df['FAF'], bins=bins, labels=labels, right=False)
        df['Activity_Level'] = df['Activity_Level'].astype(str)
        
    # Water Intake Categories (CH2O)
    if 'CH2O' in df.columns:
        bins = [0, 1.5, 2.5, 4]
        labels = ['Low', 'Medium', 'High']
        df['Water_Intake_Category'] = pd.cut(df['CH2O'], bins=bins, labels=labels, right=False)
        df['Water_Intake_Category'] = df['Water_Intake_Category'].astype(str)
        
    return df

def main():
    """
    Main function to train and evaluate machine learning models for the AI Diet Recommendation System.
    """
    print("========================================")
    print("1 & 2. Loading and Analyzing the Dataset")
    print("========================================")
    
    data_path = "datasets/obesity.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    print("\nDataset Shape before engineering:", df.shape)
    
    # Apply Feature Engineering
    df = feature_engineering(df)
    print("\nDataset Shape after engineering:", df.shape)
    
    print("\n========================================")
    print("3. Identifying Column Types")
    print("========================================")
    
    target_col = 'NObeyesdad'
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        
    print("\nNumerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)
    
    print("\n========================================")
    print("4. Preprocessing the Dataset")
    print("========================================")
    
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col])
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining features shape (X_train):", X_train.shape)
    print("Testing features shape (X_test):", X_test.shape)
    
    print("\n========================================")
    print("5 & 6. Training, Cross-Validation, and Evaluating Models")
    print("========================================")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss')
    }
    
    results = {}
    best_model_name = ""
    best_f1_score = -1
    best_model = None
    
    for name, model in models.items():
        print(f"\nTraining and evaluating {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        cv_mean = cv_scores.mean()
        
        # Train on full train set
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'CV Score': cv_mean,
            'Model': model
        }
        
        print(f"Results for {name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  CV Score:  {cv_mean:.4f}")
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_model = model
            
    print("\n========================================")
    print("7. Selecting Best Model")
    print("========================================")
    print(f"\nThe best performing model is >> {best_model_name} << with an F1 Score of {best_f1_score:.4f}")
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n========================================")
    print("8 & 9. Saving Model and Preprocessors")
    print("========================================")
    
    model_path = os.path.join(models_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nBest model ({best_model_name}) saved successfully to '{model_path}'")
    
    target_encoder_path = os.path.join(models_dir, "target_encoder.pkl")
    with open(target_encoder_path, "wb") as f:
        pickle.dump(target_encoder, f)
    print(f"Target encoder saved to '{target_encoder_path}'")
        
    feature_names_path = os.path.join(models_dir, "feature_names.pkl")
    with open(feature_names_path, "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)
    print(f"Feature names saved to '{feature_names_path}'")
    
    print("\n========================================")
    print("10. Generating Feature Importance Chart")
    print("========================================")
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        features = X_train.columns
        
        feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)
        
        print("\nTop 15 Feature Importances:")
        print(feat_imp_df.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, hue='Feature', dodge=False, palette='viridis', legend=False)
        plt.title(f'Top 15 Feature Importances ({best_model_name})')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        chart_path = os.path.join(models_dir, "feature_importance.png")
        plt.savefig(chart_path)
        print(f"\nFeature importance chart saved successfully to '{chart_path}'")
    else:
        print(f"\nModel {best_model_name} does not support feature_importances_ attribute.")
        
    print("\n========================================")
    print("Training pipeline completed successfully.")
    print("========================================")

if __name__ == "__main__":
    main()