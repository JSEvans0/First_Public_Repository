import streamlit as st
import pandas as pd
import plotly.express as px


#  # Title
#  st.title("My First Steamlit App")
#  
#  # Write some text
#  st.write("Welcome to my first Streamlet app!")
#  



# Sidebar for navigation
page = st.sidebar.selectbox("Select a Page", ["Home","Data Dashboard","About"])

# Displat different pages based on selection
if page == "Home":
    st.title("Welcome to the Home Page")
    st.write("This is the home page of your multi-page app.")

    # Add a slider

    number = st.slider("Pick a number", 10, 99, 50)

    # Display the result
    st.write(f"You selected: {number}")

    # Color Picker Test
    st.color_picker("Red Display", str(f"#{number}0000"))


elif page == "Data Dashboard":
    st.title("Data Dashboard")
    # st.write("Here's where your data visualization and dashboards will go.")

    # DataFrame
    data = pd.DataFrame({
        "Catagory": ["A", "B", "C", "D"],
        "Values": [10, 23, 45, 15]
    })

    st.write("Here's the sample data:")
    st.dataframe(data)

    # Plotly Chart
    fig = px.bar(data, x = "Catagory", y = "Values", title = "Category Values",
                 labels = {"Catagory":"Catagory", "Values":"Values"})
    
    # Displat Chart
    st.plotly_chart(fig)

elif page == "About":
    st.title("About")
    st.write("This is an example of a multi-page Streamlit app.")


#  # Title
#  st.title("Simple Data Dashboard")
#  
#  # Add sliders for each category with default values
#  value_a = st.slider("Value for Category A", 0, 100, 47)
#  value_b = st.slider("Value for Category B", 0, 100, 41)
#  value_c = st.slider("Value for Category C", 0, 100, 64)
#  value_d = st.slider("Value for Category D", 0, 100, 50)
#  
#  # Create a sample DataFrame
#  updated_data = pd.DataFrame({
#      "Catagory": ["A", "B", "C", "D"],
#      "Values": [value_a, value_b, value_c, value_d]
#  })
#  
#  # Write text and display DataFrame
#  # st.write("Here's the sample data:")
#  # st.dataframe(data)
#  
#  
#  
#  # Create a bar plot using Plotly
#  fig = px.bar(updated_data, x = "Catagory", y = "Values", title = "Updated Category Values",
#               labels = {"Catagoty":"Catagoty", "Values":"Values"})
#  
#  # Display the plot in Streamlit
#  st.plotly_chart(fig)