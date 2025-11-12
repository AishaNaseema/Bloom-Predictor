import streamlit as st
import joblib
import numpy as np

# PAGE CONFIG

st.set_page_config(page_title="🌸 Petal Predictor", page_icon="🌼", layout="centered")

# CUSTOM PAGE STYLING
page_bg = """
<style>
body {
    background: linear-gradient(135deg, #FDEFF9 0%, #ECF4FF 100%);
    color: #222;
    font-family: 'Trebuchet MS', sans-serif;
}

h1 {
    text-align: center;
    color: #D63384;
    font-size: 2.5em;
}

.stButton>button {
    background-color: #D63384;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 1.1em;
}

.stButton>button:hover {
    background-color: #C2185B;
}

footer, .stDeployButton {
    display: none !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# APP HEADER
st.title("🌼 Petal Predictor")
st.write("Enter the flower’s measurements to predict its species 🌸")

# LOAD MODEL & ENCODER
model = joblib.load('iris_model.pkl')
le = joblib.load('label_encoder.pkl')

# USER INPUT
st.subheader("🌺 Flower Measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# PREDICTION
if st.button("Predict 🌸"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_species = le.inverse_transform(prediction)[0]

    # 🌸 Choose emoji or image based on species
    if predicted_species == "Iris-setosa":
        flower_emoji = "🌷"
        flower_name = "Setosa — the Small Elegant Bloom"
    elif predicted_species == "Iris-versicolor":
        flower_emoji = "🌻"
        flower_name = "Versicolor — the Vibrant Mid-size Iris"
    else:
        flower_emoji = "🌹"
        flower_name = "Virginica — the Grand Bloom"

    st.success(f"{flower_emoji} **{flower_name}**")
    st.write(f"Predicted Species: **{predicted_species}**")


# FOOTER
st.write("---")
st.caption("Built with ❤️ using Streamlit + Scikit-learn + Python")

