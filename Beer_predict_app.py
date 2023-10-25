import streamlit as st
import pandas as pd
import joblib 

st.title('Beer Type Prediction Project')

# Load the KNN model
model = joblib.load('models/knn_pipeline_compression_7.joblib')

# Load brewery names
with open('models/brewery_names.pkl', 'rb') as file:
    brewery_names = pd.read_pickle(file)

print(brewery_names)


# Get user inputs
    
brewery_name = st.selectbox('Brewery Name', brewery_names)
review_aroma = st.number_input('Input Aroma Rating (1-5)')
review_appearance = st.number_input('Input Appearance Rating (1-5)')
review_palate = st.number_input('Input Palate Rating (1-5)')
review_taste = st.number_input('Input Taste Rating (1-5)')
beer_abv = st.number_input('Input Beer Abv (0-20)')

# Create a predict button
if st.button('Predict Beer Type'):
    # Prepare the data for prediction
    new_data = {
        'brewery_name': brewery_name,
        'review_aroma': review_aroma,
        'review_appearance': review_appearance,
        'review_palate': review_palate,
        'review_taste': review_taste,
        'beer_abv': beer_abv
    }

    # Create a DataFrame from the new data
    new_data_df = pd.DataFrame([new_data])

    # Make a prediction using the pre-trained KNN model
    prediction = model.predict(new_data_df)

    # Display the predicted beer style
    st.write(f'<h2>Predicted Beer Type: {prediction[0]}</h2>', unsafe_allow_html=True)




