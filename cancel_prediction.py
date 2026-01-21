<<<<<<< HEAD
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# Load Model & Encoders

model = joblib.load(r"D:\Interview\rf_cancellation_model.pkl")
encoders = joblib.load(r"D:\Interview\encoder.pkl")

st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="centered")
st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Predict whether a hotel booking is likely to be canceled")


# Sidebar Input Form

st.sidebar.header("Booking Information")

# Numeric Inputs
lead_time = st.sidebar.number_input("Lead Time (days before arrival)", 0, 365, 30)
adr = st.sidebar.number_input("Average Daily Rate (ADR)", 0.0, 500.0, 80.0)
booking_changes = st.sidebar.number_input("Booking Changes", 0, 10, 0)
previous_cancellations = st.sidebar.number_input("Previous Cancellations", 0, 10, 0)
days_in_waiting_list = st.sidebar.number_input("Days in Waiting List", 0, 200, 0)
total_of_special_requests = st.sidebar.number_input("Special Requests", 0, 5, 0)
required_car_parking_spaces = st.sidebar.number_input("Car Parking Spaces", 0, 3, 0)

# Guests & Nights
adults = st.sidebar.number_input("Adults", 1, 10, 2)
children = st.sidebar.number_input("Children", 0, 5, 0)
babies = st.sidebar.number_input("Babies", 0, 5, 0)
total_guests = adults + children + babies

stays_in_weekend_nights = st.sidebar.number_input("Weekend Nights", 0, 10, 0)
stays_in_week_nights = st.sidebar.number_input("Week Nights", 1, 30, 2)
total_nights = stays_in_weekend_nights + stays_in_week_nights

# Boolean
is_repeated_guest = st.sidebar.selectbox("Repeated Guest", ["No", "Yes"])
is_repeated_guest = 1 if is_repeated_guest == "Yes" else 0

# Categorical Inputs (display label)
hotel = st.sidebar.selectbox("Hotel Type", encoders['hotel'].classes_)
market_segment = st.sidebar.selectbox("Market Segment", encoders['market_segment'].classes_)
distribution_channel = st.sidebar.selectbox("Distribution Channel", encoders['distribution_channel'].classes_)
deposit_type = st.sidebar.selectbox("Deposit Type", encoders['deposit_type'].classes_)
customer_type = st.sidebar.selectbox("Customer Type", encoders['customer_type'].classes_)


# Build Input DataFrame
input_data = pd.DataFrame({
    'hotel':[hotel],
    'lead_time':[lead_time],
    'adr':[adr],
    'booking_changes':[booking_changes],
    'previous_cancellations':[previous_cancellations],
    'days_in_waiting_list':[days_in_waiting_list],
    'total_of_special_requests':[total_of_special_requests],
    'required_car_parking_spaces':[required_car_parking_spaces],
    'is_repeated_guest':[is_repeated_guest],
    'market_segment':[market_segment],
    'distribution_channel':[distribution_channel],
    'deposit_type':[deposit_type],
    'customer_type':[customer_type],
    'total_night':[total_nights],
    'total_guest':[total_guests]
})

# Encode categorical columns for model
cat_cols = ['hotel','market_segment','distribution_channel','deposit_type','customer_type']
for col in cat_cols:
    input_data[col] = encoders[col].transform(input_data[col])

# Ensure feature order matches model
FEATURE_ORDER = model.feature_names_in_
input_data = input_data[FEATURE_ORDER]


# Prediction
st.subheader("Prediction Result")
if st.button("Predict Cancellation Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    prediction_label = "Canceled" if prediction==1 else "Not Canceled"

    st.write(f"### Booking Status: **{prediction_label}**")
    st.write(f"Cancellation Probability: **{probability:.2f}**")

    if prediction==1:
        st.error("⚠️ High Risk of Cancellation")
        st.write("- Send reminder email before arrival")
        st.write("- Offer small check-in incentive")
        st.write("- Request reconfirmation closer to arrival date")
    else:
        st.success("✅ Low Risk of Cancellation")
        st.write("- Prioritize room allocation")
        st.write("- Offer loyalty benefit for repeat visit")


# Show Mapping of Encoders
with st.expander("ℹ️ Category Mapping (LabelEncoder)"):
    for col, le in encoders.items():
        st.write(f"**{col}**")
        for i, label in enumerate(le.classes_):
            st.write(f"{i} → {label}")
        st.markdown("---")

st.caption("Hotel Booking Cancellation Prediction Dashboard")
=======
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# Load Model & Encoders

model = joblib.load(r"D:\Interview\rf_cancellation_model.pkl")
encoders = joblib.load(r"D:\Interview\encoder.pkl")

st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="centered")
st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Predict whether a hotel booking is likely to be canceled")


# Sidebar Input Form

st.sidebar.header("Booking Information")

# Numeric Inputs
lead_time = st.sidebar.number_input("Lead Time (days before arrival)", 0, 365, 30)
adr = st.sidebar.number_input("Average Daily Rate (ADR)", 0.0, 500.0, 80.0)
booking_changes = st.sidebar.number_input("Booking Changes", 0, 10, 0)
previous_cancellations = st.sidebar.number_input("Previous Cancellations", 0, 10, 0)
days_in_waiting_list = st.sidebar.number_input("Days in Waiting List", 0, 200, 0)
total_of_special_requests = st.sidebar.number_input("Special Requests", 0, 5, 0)
required_car_parking_spaces = st.sidebar.number_input("Car Parking Spaces", 0, 3, 0)

# Guests & Nights
adults = st.sidebar.number_input("Adults", 1, 10, 2)
children = st.sidebar.number_input("Children", 0, 5, 0)
babies = st.sidebar.number_input("Babies", 0, 5, 0)
total_guests = adults + children + babies

stays_in_weekend_nights = st.sidebar.number_input("Weekend Nights", 0, 10, 0)
stays_in_week_nights = st.sidebar.number_input("Week Nights", 1, 30, 2)
total_nights = stays_in_weekend_nights + stays_in_week_nights

# Boolean
is_repeated_guest = st.sidebar.selectbox("Repeated Guest", ["No", "Yes"])
is_repeated_guest = 1 if is_repeated_guest == "Yes" else 0

# Categorical Inputs (display label)
hotel = st.sidebar.selectbox("Hotel Type", encoders['hotel'].classes_)
market_segment = st.sidebar.selectbox("Market Segment", encoders['market_segment'].classes_)
distribution_channel = st.sidebar.selectbox("Distribution Channel", encoders['distribution_channel'].classes_)
deposit_type = st.sidebar.selectbox("Deposit Type", encoders['deposit_type'].classes_)
customer_type = st.sidebar.selectbox("Customer Type", encoders['customer_type'].classes_)


# Build Input DataFrame
input_data = pd.DataFrame({
    'hotel':[hotel],
    'lead_time':[lead_time],
    'adr':[adr],
    'booking_changes':[booking_changes],
    'previous_cancellations':[previous_cancellations],
    'days_in_waiting_list':[days_in_waiting_list],
    'total_of_special_requests':[total_of_special_requests],
    'required_car_parking_spaces':[required_car_parking_spaces],
    'is_repeated_guest':[is_repeated_guest],
    'market_segment':[market_segment],
    'distribution_channel':[distribution_channel],
    'deposit_type':[deposit_type],
    'customer_type':[customer_type],
    'total_night':[total_nights],
    'total_guest':[total_guests]
})

# Encode categorical columns for model
cat_cols = ['hotel','market_segment','distribution_channel','deposit_type','customer_type']
for col in cat_cols:
    input_data[col] = encoders[col].transform(input_data[col])

# Ensure feature order matches model
FEATURE_ORDER = model.feature_names_in_
input_data = input_data[FEATURE_ORDER]


# Prediction
st.subheader("Prediction Result")
if st.button("Predict Cancellation Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    prediction_label = "Canceled" if prediction==1 else "Not Canceled"

    st.write(f"### Booking Status: **{prediction_label}**")
    st.write(f"Cancellation Probability: **{probability:.2f}**")

    if prediction==1:
        st.error("⚠️ High Risk of Cancellation")
        st.write("- Send reminder email before arrival")
        st.write("- Offer small check-in incentive")
        st.write("- Request reconfirmation closer to arrival date")
    else:
        st.success("✅ Low Risk of Cancellation")
        st.write("- Prioritize room allocation")
        st.write("- Offer loyalty benefit for repeat visit")


# Show Mapping of Encoders
with st.expander("ℹ️ Category Mapping (LabelEncoder)"):
    for col, le in encoders.items():
        st.write(f"**{col}**")
        for i, label in enumerate(le.classes_):
            st.write(f"{i} → {label}")
        st.markdown("---")

st.caption("Hotel Booking Cancellation Prediction Dashboard")
>>>>>>> 869c92712718e54fc1306771dcac0139738c8e73
