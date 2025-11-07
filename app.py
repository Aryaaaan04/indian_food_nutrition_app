import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image

# -------------------
# Load YOLO Model
# -------------------
model_path = "best.pt"  # your model file (upload or link from Drive)
model = YOLO(model_path)

# -------------------
# Load Nutrient Data
# -------------------
excel_path = "Anuvaad_INDB_2024.11.xlsx"
nutrients = pd.read_excel(excel_path)

# -------------------
# Function to Get Nutrient Info
# -------------------
def get_nutrients(food_name):
    match = nutrients[nutrients["food_name"].str.contains(food_name, case=False, na=False)]
    if not match.empty:
        avg_data = match.mean(numeric_only=True)
        return {
            "Calories (kcal)": round(avg_data.get("energy_kcal", 0), 2),
            "Protein (g)": round(avg_data.get("protein_g", 0), 2),
            "Carbs (g)": round(avg_data.get("carb_g", 0), 2),
            "Fat (g)": round(avg_data.get("fat_g", 0), 2),
            "Fibre (g)": round(avg_data.get("fibre_g", 0), 2),
            "Matched Rows": len(match)
        }
    else:
        return None

# -------------------
# Streamlit App UI
# -------------------
st.title("🍛 Indian Food Detection & Nutrition Estimation App")
st.write("Upload an Indian food image to detect its name and view its nutritional values.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prediction
    st.write("🔍 Detecting food type...")
    results = model.predict(np.array(image))

    if results[0].probs is not None:
        class_id = results[0].probs.top1
        class_name = results[0].names[class_id]
        confidence = results[0].probs.top1conf.item() * 100

        st.success(f"✅ Predicted Food: **{class_name}** ({confidence:.2f}% confidence)")

        # Get nutrients
        info = get_nutrients(class_name)
        if info:
            st.subheader("🥗 Nutritional Information (per 100g approx)")
            st.table(pd.DataFrame([info]))
        else:
            st.warning("⚠️ No nutritional data found for this dish in database.")
    else:
        st.error("❌ No food detected. Try another image.")
