import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

print("🤖 INITIALIZING AI BRAIN TRAINING ENGINE (XGBOOST)...")

processed_dir = "processed"
dataset_path = os.path.join(processed_dir, "global_multi_country_ml_dataset.csv")

# 1. LOAD THE GLOBAL SIMULATION DATASET WRITTEN IN STEP 3
if not os.path.exists(dataset_path):
    print(f"❌ ERROR: Could not find the global dataset file at {dataset_path}!")
    print("Please make sure step 03 ran completely and saved the file.")
    exit()

df = pd.read_csv(dataset_path)

print(f"📊 Successfully loaded global dataset framework containing {len(df)} simulation worlds.")

# One-hot encode the 'country' text column so the math model can understand it
df_encoded = pd.get_dummies(df, columns=['country'], drop_first=False)

# 2. SEPARATE INPUT FEATURES FROM TARGET PREDICTIONS
# Features (What the AI uses to guess)
X = df_encoded.drop(columns=['scenario_id', 'predicted_total_cases', 'predicted_peak_day', 'predicted_max_daily_cases'], errors='ignore')

# Targets (What we want the AI to predict)
y_total_cases = df_encoded['predicted_total_cases']
y_peak_day = df_encoded['predicted_peak_day']

# 3. SPLIT INTO TRAINING AND TESTING SETS (80% to learn, 20% to test its smarts)
X_train, X_test, y_train_cases, y_test_cases = train_test_split(X, y_total_cases, test_size=0.2, random_state=42)
_, _, y_train_peak, y_test_peak = train_test_split(X, y_peak_day, test_size=0.2, random_state=42)

print(f"🧠 Training set size: {X_train.shape[0]} rows | Testing verification set size: {X_test.shape[0]} rows")

# 4. TRAIN AI MODEL 1: PREDICTING TOTAL OUTBREAK CASES
print("🏋️‍♂️ Training AI Model 1 of 2: Learning to predict cumulative infection totals...")
model_cases = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
model_cases.fit(X_train, y_train_cases)

# Evaluate Model 1
preds_cases = model_cases.predict(X_test)
mape_cases = mean_absolute_percentage_error(y_test_cases, preds_cases) * 100
r2_cases = r2_score(y_test_cases, preds_cases)

# 5. TRAIN AI MODEL 2: PREDICTING THE OUTBREAK PEAK DAY
print("🏋️‍♂️ Training AI Model 2 of 2: Learning to predict outbreak peak day timelines...")
model_peak = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
model_peak.fit(X_train, y_train_peak)

# Evaluate Model 2
preds_peak = model_peak.predict(X_test)
mape_peak = mean_absolute_percentage_error(y_test_peak, preds_peak) * 100
r2_peak = r2_score(y_test_peak, preds_peak)

# 6. PRINT METRICS SHOWING HOW GOOD YOUR PROJECT IS
print("\n========================================================")
print("🎯 AI MODEL PERFORMANCE SUMMARY REPORT")
print("========================================================")
print(f"📈 TOTAL CASES MODEL   -> Accuracy (R² Score): {r2_cases*100:.2f}% | Margin of Error: {mape_cases:.2f}%")
print(f"📅 PEAK TIMELINE MODEL -> Accuracy (R² Score): {r2_peak*100:.2f}% | Margin of Error: {mape_peak:.2f}%")
print("========================================================")

# 7. CHOOSE A RANDOM TEST ROW TO SHOWCASE AI PREDICTION VS REALITY
random_index = np.random.choice(X_test.index)
single_test_features = X_test.loc[[random_index]]

actual_cases = y_test_cases.loc[random_index]
ai_predicted_cases = int(model_cases.predict(single_test_features)[0])

print("\n🔮 LIVE MODEL SHOWCASE (AI BLIND PREDICTION TEST):")
print(f"   ↳ [Actual Simulation Truth]: Total Cases = {actual_cases:,}")
print(f"   ↳ [XGBoost Smart Prediction]: Total Cases = {ai_predicted_cases:,}")
print(f"   ↳ Difference Variance      : {abs(actual_cases - ai_predicted_cases):,} cases")

print("\n🎉 GRAND SUCCESS! YOUR GLOBAL EPIDEMIC MACHINE LEARNING SYSTEM IS FULLY OPERATIONAL!")