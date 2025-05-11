# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import json
import pandas as pd
import time
import base64
import os

# Configuration (must match your training)
IMG_SIZE = (224, 224)
CLASS_NAMES = {0: 'benign', 1: 'malignant', 2: 'normal'}

# Set page config
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
def set_dark_theme():
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Containers */
    .main-container {
        background-color: #1e1e1e;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        border: 1px solid #333;
    }
    
    /* Diagnosis cards */
    .diagnosis-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border-left: 6px solid;
        background-color: #2a2a2a;
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3a5a78;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4a7ba8;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Input fields */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: #2a2a2a;
        color: white;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #444;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #2a2a2a;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        background-color: #2a2a2a;
        color: #aaa;
        border: 1px solid #444;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3a3a3a;
        color: white;
        font-weight: bold;
        border-bottom: 1px solid #3a3a3a;
    }
    
    /* Custom severity indicators */
    .severity-high {
        color: #ff6b6b;
        font-weight: bold;
    }
    .severity-medium {
        color: #feca57;
        font-weight: bold;
    }
    .severity-low {
        color: #1dd1a1;
        font-weight: bold;
    }
    
    /* Text colors */
    .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Camera input */
    .stCameraInput>div>div {
        border: 2px solid #444;
        border-radius: 10px;
    }
    
    /* File uploader */
    .stFileUploader>div>div {
        border: 2px dashed #444;
        border-radius: 10px;
        background-color: #2a2a2a;
    }
    
    /* Progress bars */
    .stProgress>div>div>div {
        background-color: #3a5a78;
    }
    
    /* Expander */
    .stExpander {
        border: 1px solid #444;
        border-radius: 10px;
    }
    .stExpander>div>div>div>div {
        background-color: #2a2a2a;
    }
    
    /* Success/Warning/Error boxes */
    .stAlert {
        background-color: rgba(26, 26, 26, 0.9);
        border: 1px solid #444;
    }
    </style>
    """, unsafe_allow_html=True)

set_dark_theme()

# Load model with verification
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/final_model.h5')
        
        # Verify model structure
        assert model.input_shape[1:] == (*IMG_SIZE, 3), "Model input shape mismatch"
        assert model.output_shape[-1] == len(CLASS_NAMES), "Model output shape mismatch"
        
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# Preprocess image to match training
def preprocess_image(img):
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = ImageOps.fit(img, IMG_SIZE, Image.LANCZOS)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        st.error(f"❌ Image processing error: {str(e)}")
        return None

# Prediction function
def predict_skin_cancer(model, img_array):
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return CLASS_NAMES[predicted_class], confidence, predictions[0]
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None, None

# Doctor recommendations with dark theme colors
def get_recommendations(prediction):
    recommendations = {
        "benign": {
            "diagnosis": "Benign Skin Lesion",
            "description": "Non-cancerous growth that doesn't spread to other parts of the body",
            "doctors": ["Dermatologist"],
            "precautions": [
                "Monitor for changes in size, shape, or color",
                "Use SPF 30+ sunscreen daily",
                "Avoid tanning beds",
                "Schedule annual skin check-ups"
            ],
            "diet": [
                "Antioxidant-rich foods (berries, leafy greens)",
                "Omega-3 fatty acids (fatty fish, flaxseeds)",
                "Stay hydrated",
                "Limit processed foods"
            ],
            "severity": "Low risk",
            "color": "#feca57",  # Yellow/Orange
            "icon": "⚠️",
            "severity_class": "severity-low"
        },
        "malignant": {
            "diagnosis": "Malignant Melanoma",
            "description": "Serious form of skin cancer that can spread to other organs",
            "doctors": ["Dermatologist", "Oncologist", "Surgeon"],
            "precautions": [
                "Seek immediate medical attention",
                "Avoid sun exposure completely",
                "Document changes with photos",
                "Do not attempt home remedies"
            ],
            "diet": [
                "High-protein foods for healing",
                "Cruciferous vegetables (broccoli, cauliflower)",
                "Turmeric and ginger for inflammation",
                "Avoid alcohol and tobacco"
            ],
            "severity": "High risk - Urgent care needed",
            "color": "#ff6b6b",  # Red
            "icon": "❗",
            "severity_class": "severity-high"
        },
        "normal": {
            "diagnosis": "Healthy Skin",
            "description": "No signs of cancerous or precancerous lesions",
            "doctors": ["Primary care physician for routine check-ups"],
            "precautions": [
                "Regular self-examinations",
                "Use sunscreen daily",
                "Stay hydrated",
                "Maintain healthy lifestyle"
            ],
            "diet": [
                "Balanced diet with fruits/vegetables",
                "Adequate protein intake",
                "Stay hydrated",
                "Limit processed foods"
            ],
            "severity": "No risk",
            "color": "#1dd1a1",  # Green
            "icon": "✅",
            "severity_class": "severity-low"
        }
    }
    
    return recommendations.get(prediction.lower(), {
        "diagnosis": "Analysis Inconclusive",
        "description": "The model couldn't confidently classify this lesion",
        "doctors": ["Dermatologist for further examination"],
        "precautions": ["Monitor for any changes", "Schedule a professional examination"],
        "diet": ["Maintain a healthy balanced diet"],
        "severity": "Unknown risk",
        "color": "#aaaaaa",  # Gray
        "icon": "❓",
        "severity_class": ""
    })

# Severity scale visualization with dark theme
def show_severity_scale(prediction, confidence):
    severity_levels = {
        "malignant": {"level": 4, "color": "#ff6b6b", "label": "High Risk"},
        "benign": {"level": 2, "color": "#feca57", "label": "Low Risk"},
        "normal": {"level": 0, "color": "#1dd1a1", "label": "No Risk"}
    }
    
    level_info = severity_levels.get(prediction.lower(), {"level": 1, "color": "#aaaaaa", "label": "Unknown"})
    
    st.markdown(f"""
    <div style="margin: 1.5rem 0;">
        <h4 style="color: #e0e0e0;">Risk Assessment</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; color: #aaa;">
            <span>No Risk</span>
            <span>Low Risk</span>
            <span>High Risk</span>
        </div>
        <div style="height: 10px; background: linear-gradient(to right, #1dd1a1, #feca57, #ff6b6b); 
                    border-radius: 5px; position: relative;">
            <div style="position: absolute; left: {level_info['level']/4*100}%; 
                        transform: translateX(-50%); top: -5px;">
                <div style="width: 0; height: 0; border-left: 8px solid transparent;
                            border-right: 8px solid transparent; border-bottom: 15px solid {level_info['color']};
                            margin: 0 auto;"></div>
                <div style="background-color: {level_info['color']}; color: #121212; padding: 2px 8px;
                            border-radius: 12px; font-size: 12px; white-space: nowrap;
                            transform: translateX(-50%); margin-top: -2px;">
                    {level_info['label']} ({confidence*100:.1f}%)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.title("🩺 DermaScan AI - Skin Cancer Detection")
    st.markdown("""
    <div style="background-color: #2a2a2a; padding: 1rem; border-radius: 10px; margin-bottom: 2rem; border: 1px solid #444;">
        Upload an image of your skin lesion for AI-powered analysis. Our system will assess whether the lesion 
        appears benign, malignant, or normal, and provide personalized recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Patient information
    with st.container():
        st.subheader("Patient Information")
        cols = st.columns(2)
        with cols[0]:
            name = st.text_input("Full Name", placeholder="John Doe")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
        with cols[1]:
            skin_type = st.selectbox("Skin Type", [
                "Type I (Very fair, always burns)",
                "Type II (Fair, usually burns)",
                "Type III (Medium, sometimes burns)",
                "Type IV (Olive, rarely burns)",
                "Type V (Brown, very rarely burns)",
                "Type VI (Dark, never burns)"
            ])
            risk_factors = st.multiselect("Risk Factors", [
                "Family history of skin cancer",
                "Frequent sun exposure",
                "History of sunburns",
                "Light skin/hair/eyes",
                "Many moles",
                "Weakened immune system"
            ])
    
    # Image upload section
    st.subheader("Skin Lesion Analysis")
    tab1, tab2 = st.tabs(["Upload Image", "Use Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image file (JPEG, PNG)", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=300)
    
    with tab2:
        camera_img = st.camera_input("Take a photo of your skin lesion", label_visibility="collapsed")
        if camera_img is not None:
            img = Image.open(camera_img)
    
    # Analysis button
    if ('img' in locals() or 'img' in globals()) and st.button("Analyze Skin Lesion", type="primary"):
        with st.spinner("Analyzing image..."):
            # Preprocess and predict
            img_array = preprocess_image(img)
            if img_array is not None:
                prediction, confidence, probs = predict_skin_cancer(model, img_array)
                
                if prediction is not None:
                    recommendations = get_recommendations(prediction)
                    
                    # Results container
                    with st.container():
                        st.subheader("Diagnosis Results")
                        
                        # Diagnosis card
                        st.markdown(f"""
                        <div class="diagnosis-card" style="border-left-color: {recommendations['color']}">
                            <div style="display: flex; align-items: center; gap: 15px;">
                                <span style="font-size: 2rem;">{recommendations['icon']}</span>
                                <div>
                                    <h2 style="margin: 0; color: {recommendations['color']}">
                                        {recommendations['diagnosis']}
                                    </h2>
                                    <p style="margin: 0.5rem 0 0; color: #aaa;">
                                        {recommendations['description']}
                                    </p>
                                </div>
                            </div>
                            <div style="margin-top: 1rem;">
                                <p style="margin: 0.5rem 0; color: #e0e0e0;">
                                    <strong>Confidence:</strong> {confidence*100:.1f}%
                                </p>
                                <p style="margin: 0.5rem 0; color: #e0e0e0;">
                                    <strong>Severity:</strong> 
                                    <span class="{recommendations['severity_class']}">
                                        {recommendations['severity']}
                                    </span>
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Severity scale
                        show_severity_scale(prediction, confidence)
                        
                        # Probability distribution
                        st.subheader("Detailed Analysis")
                        prob_df = pd.DataFrame({
                            "Condition": [CLASS_NAMES[i].capitalize() for i in range(len(CLASS_NAMES))],
                            "Probability (%)": [f"{p*100:.1f}" for p in probs]
                        }).sort_values("Probability (%)", ascending=False)
                        
                        # Custom styling for the dataframe
                        st.dataframe(
                            prob_df.style.applymap(lambda x: 'color: white').background_gradient(
                                cmap='Blues', subset=['Probability (%)']
                            ),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Recommendations
                        st.subheader("Personalized Recommendations")
                        cols = st.columns(2)
                        with cols[0]:
                            with st.container():
                                st.markdown("##### 👨‍⚕️ Recommended Specialists")
                                for doc in recommendations['doctors']:
                                    st.markdown(f"- {doc}")
                                
                                st.markdown("##### 🛡️ Precautions")
                                for prec in recommendations['precautions']:
                                    st.markdown(f"- {prec}")
                        
                        with cols[1]:
                            with st.container():
                                st.markdown("##### 🍎 Dietary Advice")
                                for diet in recommendations['diet']:
                                    st.markdown(f"- {diet}")
                                
                                st.markdown("##### 🔍 Next Steps")
                                if prediction == "malignant":
                                    st.error("**Urgent:** Seek immediate medical attention from a dermatologist or oncologist")
                                elif prediction == "benign":
                                    st.warning("**Recommended:** Schedule an appointment with a dermatologist for confirmation")
                                else:
                                    st.success("**Maintenance:** Continue regular skin self-examinations")
                        
                        # Report generation
                        st.subheader("Download Report")
                        report_text = f"""
                        DERMASCAN AI - SKIN LESION ANALYSIS REPORT
                        {'='*60}
                        
                        PATIENT INFORMATION
                        {'-'*60}
                        Name: {name}
                        Age: {age}
                        Skin Type: {skin_type}
                        Risk Factors: {', '.join(risk_factors) if risk_factors else 'None'}
                        
                        DIAGNOSIS
                        {'-'*60}
                        Condition: {recommendations['diagnosis']}
                        Confidence Level: {confidence*100:.1f}%
                        Description: {recommendations['description']}
                        Severity: {recommendations['severity']}
                        
                        DETAILED ANALYSIS
                        {'-'*60}
                        {prob_df.to_string(index=False)}
                        
                        RECOMMENDATIONS
                        {'-'*60}
                        Specialists to Consult:
                        {chr(10).join('- ' + d for d in recommendations['doctors'])}
                        
                        Precautions:
                        {chr(10).join('- ' + p for p in recommendations['precautions'])}
                        
                        Dietary Advice:
                        {chr(10).join('- ' + d for d in recommendations['diet'])}
                        
                        NEXT STEPS
                        {'-'*60}
                        {recommendations['severity']} - {recommendations['icon']} {recommendations['diagnosis']}
                        
                        REPORT GENERATED ON: {time.strftime('%Y-%m-%d %H:%M:%S')}
                        """
                        
                        st.download_button(
                            "📄 Download Full Report",
                            report_text,
                            file_name=f"DermaScan_Report_{name.replace(' ', '_')}_{time.strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            type="primary"
                        )
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **Disclaimer:** This AI tool provides preliminary analysis only and is not a substitute for professional medical advice. 
    Always consult a qualified healthcare provider for any concerns about your skin health. The accuracy of this tool 
    depends on image quality and may not detect all skin conditions.
    """)

if __name__ == "__main__":
    main()