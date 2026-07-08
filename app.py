import streamlit as st
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI Text Authenticity Detector",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>

.stApp{
    background:#0F172A;
}

h1,h2,h3{
    color:white;
    text-align:center;
}

p,label{
    color:#CBD5E1 !important;
    font-size:16px;
}

.stTextArea textarea{
    background:#1E293B !important;
    color:white !important;
    border:1px solid #334155 !important;
    border-radius:12px !important;
    font-size:16px !important;
}

.stButton>button{
    width:100%;
    height:52px;
    background:#2563EB;
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:600;
}

.stButton>button:hover{
    background:#1D4ED8;
}

div.stProgress > div > div > div{
    background:#2563EB;
}

hr{
    border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🤖 AI Text Authenticity Detector")

st.write(
    "Analyze text and determine whether the provided text is written by a human or generated using AI."
)

st.divider()

# -------------------------------------------------
# Input
# -------------------------------------------------
user_input = st.text_area(
    "Input Text",
    placeholder="Paste your text here...",
    height=220
)

# -------------------------------------------------
# Statistics
# -------------------------------------------------
if user_input.strip():

    words = len(user_input.split())
    chars = len(user_input)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="
        background:#1E293B;
        border:1px solid #334155;
        border-radius:12px;
        padding:20px;
        text-align:center;">

        <div style="
        color:#94A3B8;
        font-size:18px;">
        📝 Words
        </div>

        <div style="
        color:#FFFFFF;
        font-size:34px;
        font-weight:bold;
        margin-top:8px;">
        {words}
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
        background:#1E293B;
        border:1px solid #334155;
        border-radius:12px;
        padding:20px;
        text-align:center;">

        <div style="
        color:#94A3B8;
        font-size:18px;">
        🔤 Characters
        </div>

        <div style="
        color:#FFFFFF;
        font-size:34px;
        font-weight:bold;
        margin-top:8px;">
        {chars}
        </div>

        </div>
        """, unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# Analyze Button
# -------------------------------------------------
analyze = st.button(
    "🔍 Analyze Text",
    use_container_width=True
)
# -------------------------------------------------
# Prediction
# -------------------------------------------------
if analyze:

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("Analyzing text..."):

            text_vector = vectorizer.transform([user_input])

            prediction = model.predict(text_vector)[0]

            probability = model.predict_proba(text_vector)[0]

            confidence = max(probability) * 100

        st.divider()

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        st.subheader("Analysis Result")

        if prediction == 0:
            st.success("✅ Human Written")
        else:
            st.error("⚠️ AI Generated")

        # -------------------------------------------------
        # Confidence Score Card
        # -------------------------------------------------

        st.markdown(f"""
        <div style="
        background:#1E293B;
        border:1px solid #334155;
        border-radius:12px;
        padding:20px;
        text-align:center;
        margin-bottom:15px;">

        <div style="
        color:#94A3B8;
        font-size:18px;">
        🎯 Confidence Score
        </div>

        <div style="
        color:#FFFFFF;
        font-size:38px;
        font-weight:bold;
        margin-top:10px;">
        {confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence / 100)

        st.divider()

        # -------------------------------------------------
        # Prediction Probability
        # -------------------------------------------------

        st.subheader("Prediction Probability")

        human = probability[0] * 100
        ai = probability[1] * 100

        col1, col2 = st.columns(2)

        # ---------------- Human ----------------

        with col1:

            st.markdown(f"""
            <div style="
            background:#1E293B;
            border:1px solid #22C55E;
            border-radius:12px;
            padding:20px;
            text-align:center;">

            <div style="
            color:#22C55E;
            font-size:18px;">
            🟢 Human
            </div>

            <div style="
            color:#FFFFFF;
            font-size:34px;
            font-weight:bold;
            margin-top:10px;">
            {human:.2f}%
            </div>

            </div>
            """, unsafe_allow_html=True)

        # ---------------- AI ----------------

        with col2:

            st.markdown(f"""
            <div style="
            background:#1E293B;
            border:1px solid #EF4444;
            border-radius:12px;
            padding:20px;
            text-align:center;">

            <div style="
            color:#EF4444;
            font-size:18px;">
            🔴 AI
            </div>

            <div style="
            color:#FFFFFF;
            font-size:34px;
            font-weight:bold;
            margin-top:10px;">
            {ai:.2f}%
            </div>

            </div>
            """, unsafe_allow_html=True)

        st.write("### Human Probability")
        st.progress(human / 100)

        st.write("### AI Probability")
        st.progress(ai / 100)

        st.divider()

        # -------------------------------------------------
        # Final Summary
        # -------------------------------------------------

        if prediction == 0:

            st.markdown("""
            <div style="
            background:#052E16;
            border:1px solid #22C55E;
            border-radius:12px;
            padding:18px;
            text-align:center;
            color:#DCFCE7;
            font-size:17px;
            font-weight:600;">

            ✅ This text is most likely written by a human.

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
            background:#450A0A;
            border:1px solid #EF4444;
            border-radius:12px;
            padding:18px;
            text-align:center;
            color:#FECACA;
            font-size:17px;
            font-weight:600;">

            ⚠️ This text is most likely generated by AI.

            </div>
            """, unsafe_allow_html=True)