import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model and dataset
@st.cache_resource
def load_model():
    return YOLO("best.pt")

@st.cache_data
def load_nutrition_data():
    return pd.read_excel("Anuvaad_INDB_2024.11.xlsx")

model = load_model()
nutrients = load_nutrition_data()

st.title("🍛 Indian Food Recognition & Nutrition Estimation")
st.markdown("Upload a food image to detect the dish and view its nutritional information!")

uploaded_file = st.file_uploader("📸 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting food item..."):
        results = model.predict(img)
        pred = results[0].probs
        if pred is not None:
            food_name = results[0].names[pred.top1]
            conf = pred.top1conf.item() * 100
            st.success(f"✅ Detected Food: **{food_name}** ({conf:.2f}% confidence)")

            # Nutrient lookup
            match = nutrients[nutrients["food_name"].str.contains(food_name, case=False, na=False)]

            if not match.empty:
                avg_data = match.mean(numeric_only=True)
                st.subheader("🥗 Nutritional Information (per serving)")
                st.write(f"**Calories:** {round(avg_data.get('energy_kcal', 0), 2)} kcal")
                st.write(f"**Protein:** {round(avg_data.get('protein_g', 0), 2)} g")
                st.write(f"**Carbs:** {round(avg_data.get('carb_g', 0), 2)} g")
                st.write(f"**Fat:** {round(avg_data.get('fat_g', 0), 2)} g")
                st.write(f"**Fibre:** {round(avg_data.get('fibre_g', 0), 2)} g")
            else:
                st.warning("⚠️ No matching nutrient data found for this item.")
        else:
            st.error("❌ No food detected. Please upload a clearer image.")
