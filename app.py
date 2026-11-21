import streamlit as st
import pandas as pd
import joblib

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Airline Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp {
    background: #0b1220;
    color: #f1f5f9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(120deg, #142b4a, #173d68);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid #284d75;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.section {
    color: #60a5fa;
    font-size: 23px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

div[data-testid="stForm"] {
    background: #111e31;
    padding: 25px;
    border: 1px solid #263c58;
    border-radius: 16px;
}

div[data-testid="stButton"] button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: 700;
    width: 100%;
}

div[data-testid="stButton"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #1d4ed8;
    color: white;
}

.result-card {
    background: #132b42;
    border: 1px solid #2563eb;
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

.result-card h2 {
    color: #60a5fa;
}

.result-card h1 {
    color: white;
    font-size: 35px;
}

.metric-card {
    background: #14243a;
    border: 1px solid #263c58;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# LOAD TRAINED MODEL
# =====================================

@st.cache_resource
def load_model():

    return joblib.load(
        "airline_satisfaction_model.pkl"
    )


try:

    model = load_model()

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# =====================================
# HEADER
# =====================================

st.markdown("""
<div class="hero">

<h1>✈️ Airline Intelligence</h1>

<p>
Passenger Satisfaction Prediction Platform
</p>

<p>
Machine Learning powered passenger experience analytics
</p>

</div>
""", unsafe_allow_html=True)


# =====================================
# INTRODUCTION
# =====================================

st.markdown("""
### Welcome to Passenger Experience Analytics

Enter passenger information and service ratings
to predict airline passenger satisfaction.
""")


# =====================================
# INPUT FORM
# =====================================

with st.form("passenger_form"):

    st.markdown(
        '<div class="section">👤 Passenger Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Passenger Age",
            min_value=1,
            max_value=100,
            value=30
        )

    with col2:

        customer_type = st.selectbox(
            "Customer Type",
            ["Loyal Customer", "disloyal Customer"]
        )

        type_of_travel = st.selectbox(
            "Type of Travel",
            ["Business travel", "Personal Travel"]
        )

    with col3:

        passenger_class = st.selectbox(
            "Travel Class",
            ["Business", "Eco", "Eco Plus"]
        )

        flight_distance = st.number_input(
            "Flight Distance",
            min_value=0,
            max_value=10000,
            value=1000
        )

    st.markdown(
        '<div class="section">⏱️ Flight Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        departure_delay = st.number_input(
            "Departure Delay (Minutes)",
            min_value=0,
            value=0
        )

    with col2:

        arrival_delay = st.number_input(
            "Arrival Delay (Minutes)",
            min_value=0.0,
            value=0.0
        )

    st.markdown(
        '<div class="section">⭐ Passenger Service Ratings</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Rate each service from 0 (lowest) to 5 (highest)."
    )

    col1, col2 = st.columns(2)

    with col1:

        departure_convenience = st.slider(
            "Departure & Arrival Convenience",
            0, 5, 3
        )

        online_booking = st.slider(
            "Ease of Online Booking",
            0, 5, 3
        )

        checkin_service = st.slider(
            "Check-in Service",
            0, 5, 3
        )

        online_boarding = st.slider(
            "Online Boarding",
            0, 5, 3
        )

        gate_location = st.slider(
            "Gate Location",
            0, 5, 3
        )

        onboard_service = st.slider(
            "Onboard Service",
            0, 5, 3
        )

        seat_comfort = st.slider(
            "Seat Comfort",
            0, 5, 3
        )

        leg_room_service = st.slider(
            "Leg Room Service",
            0, 5, 3
        )

    with col2:

        cleanliness = st.slider(
            "Cleanliness",
            0, 5, 3
        )

        food_and_drink = st.slider(
            "Food and Drink",
            0, 5, 3
        )

        inflight_service = st.slider(
            "Inflight Service",
            0, 5, 3
        )

        inflight_wifi = st.slider(
            "Inflight WiFi Service",
            0, 5, 3
        )

        inflight_entertainment = st.slider(
            "Inflight Entertainment",
            0, 5, 3
        )

        baggage_handling = st.slider(
            "Baggage Handling",
            0, 5, 3
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "✈️ PREDICT PASSENGER SATISFACTION"
    )


# =====================================
# PREDICTION
# =====================================

if submitted:

    input_data = pd.DataFrame([{

        "gender": gender,
        "age": age,
        "customer_type": customer_type,
        "type_of_travel": type_of_travel,
        "class": passenger_class,

        "flight_distance": flight_distance,
        "departure_delay": departure_delay,
        "arrival_delay": arrival_delay,

        "departure_and_arrival_time_convenience":
            departure_convenience,

        "ease_of_online_booking": online_booking,
        "checkin_service": checkin_service,
        "online_boarding": online_boarding,
        "gate_location": gate_location,
        "onboard_service": onboard_service,
        "seat_comfort": seat_comfort,
        "leg_room_service": leg_room_service,
        "cleanliness": cleanliness,
        "food_and_drink": food_and_drink,
        "inflight_service": inflight_service,
        "inflight_wifi_service": inflight_wifi,
        "inflight_entertainment": inflight_entertainment,
        "baggage_handling": baggage_handling

    }])

    try:

        prediction = model.predict(input_data)[0]

        # Get prediction probability
        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)

        predicted_probability = probabilities[
            classes.index(prediction)
        ] * 100

        # Result section

        st.markdown("---")

        st.markdown(
            '<div class="section">📊 Prediction Results</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.success("😊 Passenger is Satisfied")

                result_text = "SATISFIED"

            else:

                st.warning(
                    "😐 Passenger is Neutral or Dissatisfied"
                )

                result_text = "NEUTRAL / DISSATISFIED"

            st.markdown(
                f"""
                <div class="result-card">
                    <h2>Predicted Satisfaction</h2>
                    <h1>{result_text}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Prediction Confidence</h3>
                    <h1>{predicted_probability:.2f}%</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                float(predicted_probability / 100)
            )

            st.caption(
                "Probability assigned to the predicted class."
            )

        # Input summary

        with st.expander("View Passenger Input Summary"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Prediction error: {e}")


# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown("""
<center>

**AIRLINE INTELLIGENCE**

Machine Learning | Passenger Analytics | Streamlit

Developed as an End-to-End Machine Learning Project

</center>
""", unsafe_allow_html=True)