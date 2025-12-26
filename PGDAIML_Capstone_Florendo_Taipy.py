import os
import joblib
import pandas as pd
from taipy.gui import Gui, notify, download # Added download import

MODELS_DIR = "models"

# --- Configurations ---
CONFIGS = {
    "recovery": {"features": {0: "Head Grade_Au gpt", 1: "Head Grade_Ag gpt", 2: "Tonnage_Processed_tons"}, "unit": "%", "color": "#1f77b4"},
    "tonnage": {"features": {0: "Off-Vein Meterage", 1: "On-Vein Meterage", 2: "Total_Meterage"}, "unit": "tons", "color": "#2ca02c"},
    "gold": {"features": {0: "Head Grade_Au gpt", 1: "Head Grade_Ag gpt", 2: "Tonnage_Processed_tons"}, "unit": "oz Au", "color": "#FFD700"},
    "silver": {"features": {0: "Head Grade_Au gpt", 1: "Head Grade_Ag gpt", 2: "Tonnage_Processed_tons"}, "unit": "oz Ag", "color": "#C0C0C0"}
}

# --- Helpers ---
def get_metadata(model_name):
    name_clean = model_name.lower()
    for key, config in CONFIGS.items():
        if key in name_clean: 
            return config, key.title()
    return {"features": {}, "unit": "", "color": "#000000"}, "Unknown"

def get_filtered_models(category):
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")])
    if category == "Baseline":
        return [f for f in all_files if "baseline" in f.lower()]
    return [f for f in all_files if "baseline" not in f.lower()]

# --- Core Functions ---

def load_model_into_state(state, model_name):
    if not model_name: return
    try:
        model = joblib.load(os.path.join(MODELS_DIR, model_name))
        n = int(model.n_features_in_)
        metadata, target_name = get_metadata(model_name)
        
        state.current_unit = metadata["unit"]
        state.line_color = metadata["color"]
        mapping = metadata["features"]
        feature_names = [mapping.get(i, f"Feature {i+1}") for i in range(n)]
        
        state.df_features = pd.DataFrame({"Feature": feature_names, "Value": [0.0] * n})
        
        if hasattr(model, "feature_importances_"):
            state.importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
        else:
            state.importance_df = pd.DataFrame(columns=["Feature", "Importance"])
            
        state.prediction = ""
        state.batch_results = pd.DataFrame(columns=["Index", "Prediction"])
        state.batch_summary = ""
        state.chart_title = f"Trend: Select Model and Upload Data"
    except Exception as e:
        notify(state, "error", f"Update failed: {str(e)}")

# --- Callbacks ---

def on_predict(state):
    try:
        model = joblib.load(os.path.join(MODELS_DIR, state.selected_model))
        values = [float(v) for v in state.df_features["Value"].tolist()]
        pred = model.predict([values])[0]
        final_val = float(pred)
        if state.current_unit == "%":
            final_val *= 100
            final_val = min(final_val, 100.0)
        state.prediction = f"{round(final_val, 2)} {state.current_unit}"
        notify(state, "success", "Calculation Successful")
    except Exception as e:
        notify(state, "error", f"Prediction error: {e}")

def on_batch_upload(state):
    if not state.batch_file: return
    try:
        df = pd.read_csv(state.batch_file, encoding='utf-8-sig')
        model = joblib.load(os.path.join(MODELS_DIR, state.selected_model))
        n_expected = int(model.n_features_in_)
        
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] < n_expected:
            notify(state, "error", f"Need {n_expected} numeric columns.")
            return
        
        input_data = numeric_df.iloc[:, :n_expected]
        raw_preds = model.predict(input_data)
        
        processed_preds = []
        for p in raw_preds:
            val = float(p)
            if state.current_unit == "%":
                val = min(val * 100, 100.0)
            processed_preds.append(round(val, 2))
            
        new_df = df.copy()
        new_df["Prediction"] = processed_preds
        new_df["Index"] = range(1, len(df) + 1)
        
        metadata, target_name = get_metadata(state.selected_model)
        state.chart_title = f"Shift Prediction Trend - {state.model_type} Model - {target_name} ({state.current_unit})"
        
        avg_val = round(sum(processed_preds) / len(processed_preds), 2)
        total_val = round(sum(processed_preds), 2)
        
        if state.current_unit == "%":
            state.batch_summary = f"Average Recovery: {avg_val}% | Min: {min(processed_preds)}% | Max: {max(processed_preds)}%"
        else:
            state.batch_summary = f"Total {target_name}: {total_val} {state.current_unit} | Average per Shift: {avg_val} {state.current_unit}"
        
        state.batch_results = new_df
        notify(state, "success", "Batch processed successfully.")
    except Exception as e:
        notify(state, "error", f"Batch error: {str(e)}")

def on_download(state):
    if state.batch_results.empty:
        notify(state, "warning", "No data to download")
        return
    # Save to a temporary CSV and trigger download
    file_path = "prediction_results.csv"
    state.batch_results.to_csv(file_path, index=False)
    download(state, content=file_path, name="Mining_Predictions.csv")
    notify(state, "info", "Downloading CSV...")

def on_category_change(state):
    state.models_list = get_filtered_models(state.model_type)
    state.selected_model = state.models_list[0] if state.models_list else ""
    load_model_into_state(state, state.selected_model)

def on_model_change(state):
    load_model_into_state(state, state.selected_model)

def on_reset(state):
    state.df_features["Value"] = 0.0
    state.prediction = ""
    state.batch_results = pd.DataFrame(columns=["Index", "Prediction"])
    state.batch_summary = ""

# --- Initial State ---
model_type = "Tuned"
models_list = get_filtered_models(model_type)
selected_model = models_list[0] if models_list else ""
df_features = pd.DataFrame(columns=["Feature", "Value"])
importance_df = pd.DataFrame(columns=["Feature", "Importance"])
batch_results = pd.DataFrame(columns=["Index", "Prediction"])
batch_summary = ""
batch_file = None
prediction = ""
current_unit = ""
chart_title = "Shift Prediction Trend"
line_color = "#FFD700"

# --- UI Layout ---
page = """
# ⛏️ ML Predictor for Gold, Silver Output and Recovery, Mine Tonnage
### Created by Jose Norbiel G. Florendo | AIM PGDAIML Capstone Project

<|layout|columns=1 1|gap=30px|
<|part|
## 📋 Selection & Single Prediction
**Category**
<|{model_type}|selector|lov={['Baseline', 'Tuned']}|on_change=on_category_change|switch=True|>

**Model Selection**
<|{selected_model}|selector|lov={models_list}|on_change=on_model_change|dropdown=True|>

<|{df_features}|table|editable=True|rebuild=True|columns=Feature;Value|type__Value=number|>

<br/>
<|layout|columns=1 1|
<|button|label=Calculate|on_action=on_predict|class_name=primary|>
<|button|label=Reset|on_action=on_reset|class_name=secondary|>
|>

<br/>
<|part|render={prediction != ""}|
### Prediction Result:
#### <|{model_type} model|text|> 
# <|{prediction}|text|>
|>
|>

<|part|
## 📊 Analysis & Batch Processing
<|part|render={not importance_df.empty}|
**Sensitivity Analysis**
<|{importance_df}|chart|type=bar|x=Feature|y=Importance|title=Feature Weight Comparison|>
|>

<hr/>

**Batch Upload (.csv)**
<|{batch_file}|file_selector|label=Upload Shift Data|on_action=on_batch_upload|extensions=.csv|>

<|part|render={not batch_results.empty}|
### <|{chart_title}|text|>
> **Batch Summary:** <|{batch_summary}|text|>

<|{batch_results}|chart|type=line|x=Index|y=Prediction|color={line_color}|>

<|layout|columns=1 1|
<|button|label=Download Results (CSV)|on_action=on_download|class_name=success|>
<|text|value= |>
|>

<br/>
<|{batch_results}|table|show_all=False|page_size=5|>
|>
|>
|>
"""

gui = Gui(page=page)

def on_init(state):
    load_model_into_state(state, state.selected_model)

if __name__ == "__main__":
    gui.run(title="Capstone Project - Mining ML Dashboard", port=5000)