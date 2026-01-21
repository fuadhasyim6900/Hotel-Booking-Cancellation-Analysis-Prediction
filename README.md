# Hotel Booking Cancellation Analysis & Prediction

This project analyzes hotel booking data to understand the factors influencing booking cancellations and predicts the likelihood of cancellations using a Random Forest model. 

## Features

- **Exploratory Data Analysis (EDA):** Explore booking patterns, seasonal trends, and customer characteristics.
- **Machine Learning Model:** Random Forest classifier to predict cancellation risk.
- **Streamlit Dashboard:** Interactive web app to input booking details and get cancellation predictions with probability.
- **Business Insights:** Recommendations to reduce cancellation rates and optimize hotel revenue.
- **Category Mapping:** Display readable labels for hotel type, market segment, distribution channel, deposit type, and customer type.

## Dataset

The dataset contains hotel reservation records, including:

- Booking and arrival details (lead time, stay duration)
- Guest characteristics (adults, children, repeated guest)
- Market and distribution channels
- Booking type and policies (deposit type, customer type)
- Average daily rate and special requests

## How to Use

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run cancel_prediction.py`
4. Input booking information and get predicted cancellation risk.

## Objective

- Understand factors driving booking cancellations
- Predict high-risk bookings
- Provide actionable recommendations to improve hotel operations and revenue management

