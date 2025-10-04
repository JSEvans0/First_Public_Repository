import streamlit as st
import pandas as pd
import plotly.express as px
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Set page title and icon
st.set_page_config(page_title="Airline Satisfaction Explorer", page_icon="✈️")

# Sidebar navigation
# page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Extras", "About"])
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!", "Conclusions/Recommendations"])

# Load dataset
df = pd.read_csv('data/cleaned_airline_passenger_satisfaction.csv')

# Home Page
if page == "Home":
    st.title("📊 Airline Satisfaction Explorer")
    st.subheader("Welcome to the Airline Satisfaction explorer app!")
    st.write("""
        This app provides an interactive platform to explore an Airline Satisfaction dataset. 
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
        Use the sidebar to navigate through the sections.
    """)
    st.image('https://media.istockphoto.com/id/1254973568/photo/empty-airport-terminal-lounge-with-airplane-on-background.jpg?s=612x612&w=0&k=20&c=WoX_hcz_igZ1NNRlwwR9Cc_EjjL4Ncf_hoTMDatg2AU=', caption="empty-airport-terminal-lounge-with-airplane-on-background")
    st.write("Use the sidebar to navigate between different sections.")


# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("""
        This dataset contains an airline passenger satisfaction survey. What factors are highly correlated to a 
        satisfied (or dissatisfied) passenger? Can you predict passenger satisfaction?
    """)
    st.image('https://media.gettyimages.com/id/168725872/photo/on-the-airplane.jpg?s=612x612&w=gi&k=20&c=0fKQ0c5Ds5wugSHi24ZSx6OQaEGBdhMMhuP-0wObMTI=', caption="on-the-airplane")

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    obj_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Satisfaction"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='satisfaction', y=b_selected_col, color='satisfaction', title=chart_title))

    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))

    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", obj_cols)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, color='satisfaction', title=chart_title))

# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns=["id", "Gender", "Customer Type", "Type of Travel", "Class", "satisfaction", "satisfaction_binary"])
    y = df['satisfaction_binary']
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

# Make Predictions Page
elif page == "Make Predictions!":
    st.title("✈️ Make Predictions")

    st.subheader("Adjust the values below to make predictions on the Airline Satisfaction dataset:")

#  0   Age                                103445 non-null  int64  
#  1   Flight Distance                    103445 non-null  int64  
#  2   Inflight wifi service              103445 non-null  int64  
#  3   Departure/Arrival time convenient  103445 non-null  int64  
#  4   Ease of Online booking             103445 non-null  int64  
#  5   Gate location                      103445 non-null  int64  
#  6   Food and drink                     103445 non-null  int64  
#  7   Online boarding                    103445 non-null  int64  
#  8   Seat comfort                       103445 non-null  int64  
#  9   Inflight entertainment             103445 non-null  int64  
#  10  On-board service                   103445 non-null  int64  
#  11  Leg room service                   103445 non-null  int64  
#  12  Baggage handling                   103445 non-null  int64  
#  13  Checkin service                    103445 non-null  int64  
#  14  Inflight service                   103445 non-null  int64  
#  15  Cleanliness                        103445 non-null  int64  
#  16  Departure Delay in Minutes         103445 non-null  int64  
#  17  Arrival Delay in Minutes           103445 non-null  float64


    # User inputs for prediction
    age = st.slider("Age", min_value=7, max_value=85, value=22)
    flight_distance = st.slider("Flight Distance", min_value=31, max_value=4000, value=1500)
    inflight_wifi_service = st.slider("Inflight Wifi Service", min_value=0, max_value=5, value=3)
    departure_arrival_time_convenient = st.slider("Departure/Arrival time convenient", min_value=0, max_value=5, value=3)
    ease_of_online_booking = st.slider("Ease of Online booking", min_value=0, max_value=5, value=3)
    gate_location = st.slider("Gate location", min_value=0, max_value=5, value=3)
    food_and_drink = st.slider("Food and drink", min_value=0, max_value=5, value=3)
    online_boarding = st.slider("Online boarding", min_value=0, max_value=5, value=3)
    seat_comfort = st.slider("Seat comfort", min_value=0, max_value=5, value=3)
    inflight_entertainment = st.slider("Inflight entertainment", min_value=0, max_value=5, value=3)
    on_board_service = st.slider("On-board service", min_value=0, max_value=5, value=3)
    leg_room_service = st.slider("Leg room service", min_value=0, max_value=5, value=3)
    baggage_handling = st.slider("Baggage handling", min_value=1, max_value=5, value=3)
    checkin_service = st.slider("Checkin service", min_value=0, max_value=5, value=3)
    inflight_service = st.slider("Inflight service", min_value=0, max_value=5, value=3)
    cleanliness = st.slider("Cleanliness", min_value=0, max_value=5, value=3)
    departure_delay_in_minutes = st.slider("Departure Delay in Minutes", min_value=0, max_value=400, value=30)
    arrival_delay_in_minutes = st.slider("Arrival Delay in Minutes", min_value=0, max_value=399, value=30)

    # User input dataframe
    user_input = pd.DataFrame({
        "Age": [age],
        "Flight Distance": [flight_distance],
        "Inflight wifi service": [inflight_wifi_service],
        "Departure/Arrival time convenient": [departure_arrival_time_convenient],
        "Ease of Online booking": [ease_of_online_booking],
        "Gate location": [gate_location],
        "Food and drink": [food_and_drink],
        "Online boarding": [online_boarding],
        "Seat comfort": [seat_comfort],
        "Inflight entertainment": [inflight_entertainment],
        "On-board service": [on_board_service],
        "Leg room service": [leg_room_service],
        "Baggage handling": [baggage_handling],
        "Checkin service": [checkin_service],
        "Inflight service": [inflight_service],
        "Cleanliness": [cleanliness],
        "Departure Delay in Minutes": [departure_delay_in_minutes],
        "Arrival Delay in Minutes": [arrival_delay_in_minutes]
    })

    st.write("### Your Input Values")
    st.dataframe(user_input)

    # Use KNN (k=9) as the model for predictions
    model = KNeighborsClassifier(n_neighbors=9)
    X = df.drop(columns=["id", "Gender", "Customer Type", "Type of Travel", "Class", "satisfaction", "satisfaction_binary"])
    y = df['satisfaction_binary']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]

    # Display the result
    st.write(f"The model predicts that the customer satisfaction rating is : **{prediction}** ({"Sastisfied" if prediction==1 else "Neutral or Dissatisfied"})")
    if prediction == 1:
        st.balloons()

elif page == "Conclusions/Recommendations":
    st.title("🔢 Conclusions/Recommendations")

    st.subheader("My Findings")
    st.write("""
        I found that "Online boarding" had the greatest positive effect on Customer satisfaction, and 
        "Departure/Arrival time convenient" had the greatest negative effect on Customer satisfaction.
        Flight distance, both Departure and Arrival delays, and Age all had next to no impact on Customer satisfaction.

        So as long as the "Online boarding" process remains positive, that should help Customer 
        satisfaction stay up, and finding a way to provide more convenient Departure/arrival times 
        would also help increase Customer satisfaction.
             
        This airline looks like they could also stand to improve their "Ease of Online booking" as well as their "Food and drink".
    """)


    X_train = ['Age', 'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes',
       'Arrival Delay in Minutes']
    
    lr = [ 8.00935979e-04,  8.33141491e-05,  4.97539465e-02, -4.24013718e-02,
       -2.09368464e-02,  1.80604006e-02, -1.27629868e-02,  1.18436966e-01,
        1.86969574e-02,  4.50229292e-02,  3.88322921e-02,  4.95985401e-02,
        1.17039679e-02,  3.29467248e-02,  4.90444882e-03,  9.79302919e-03,
        3.45975453e-04, -9.11799143e-04]


    coef_df = pd.DataFrame({
    "Feature": X_train,
    "Coef" : lr
    })

    sorted_coef_df = coef_df.sort_values(by = "Coef", ascending = True)



    st.plotly_chart(px.bar(sorted_coef_df, x="Coef", y="Feature", title="Features and Coefficients from the Linear Regression Model"))
    
    st.dataframe(coef_df.sort_values(by = "Coef", ascending = False))