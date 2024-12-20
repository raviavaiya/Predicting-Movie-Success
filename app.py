import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define a function to preprocess the input data
def preprocess_input(input_data, label_encoders, scaler):
    # Apply label encoding for categorical columns
    input_data['color'] = label_encoders['color'].transform(input_data['color'])
    input_data['genres'] = label_encoders['genres'].transform(input_data['genres'])
    input_data['content_rating'] = label_encoders['content_rating'].transform(input_data['content_rating'])
    
    # Select numerical columns and scale them
    numerical_data = input_data[['duration', 'budget', 'gross', 'num_voted_users', 
                                 'num_user_for_reviews', 'num_critic_for_reviews', 'title_year']]
    scaled_data = scaler.transform(numerical_data)
    
    # Add encoded categorical features back to the scaled data
    categorical_data = input_data[['color', 'genres', 'content_rating']].values
    final_data = np.hstack((scaled_data, categorical_data))
    
    return final_data

# Update the train_model function to read the dataset from a file
def train_model():
    # Load your actual dataset from a CSV file
    df = pd.read_csv("Dataset/movie_metadata.csv")  # Replace with the path to your CSV file

    # Categorizing the target variable based on the 'imdb_score'
    bins = [1, 3, 6, 10]
    labels = ['FLOP', 'AVG', 'HIT']
    df['box_office_status'] = pd.cut(df['imdb_score'], bins=bins, labels=labels)

    # Drop rows with missing values (NaN)
    df.dropna(inplace=True)

    # Drop unnecessary columns 'movie_title' and 'movie_imdb_link'
    df.drop(columns=['movie_title', 'movie_imdb_link'], inplace=True)

    # Assuming that 'box_office_status' is the target variable
    X_train = df[['duration', 'budget', 'gross', 'num_voted_users', 'num_user_for_reviews', 
                  'num_critic_for_reviews', 'title_year', 'color', 'genres', 'content_rating']]
    y_train = df['box_office_status']  # This is the target variable

    # Label Encoding categorical columns
    label_encoders = {
        'color': LabelEncoder(),
        'genres': LabelEncoder(),
        'content_rating': LabelEncoder()
    }
    for col in label_encoders.keys():
        X_train[col] = label_encoders[col].fit_transform(X_train[col])

    # Scaling numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[['duration', 'budget', 'gross', 'num_voted_users', 
                                                   'num_user_for_reviews', 'num_critic_for_reviews', 'title_year']])

    # Add categorical features back after encoding
    X_train_scaled = np.hstack((X_train_scaled, X_train[['color', 'genres', 'content_rating']].values))

    # Create a RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

    # Train the model
    classifier.fit(X_train_scaled, y_train)

    return classifier, label_encoders, scaler

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center;'>Movie Classification</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>This application predicts if a movie is a HIT, AVERAGE, or FLOP based on various features.</p>", 
    unsafe_allow_html=True
)

# Train the model when the app starts
classifier, label_encoders, scaler = train_model()

# Create a grid layout for inputs
with st.form("movie_form"):
    st.subheader("Enter Movie Details:")  
    
    # Movie input form
    col1, col2, col3 = st.columns(3)
    with col1:
        movie_title = st.text_input("Movie Title")
        duration = st.number_input("Duration (in minutes)", min_value=1)
        director_name = st.text_input("Director's Name")
    with col2:
        actor_1_name = st.text_input("Actor 1's Name")
        actor_2_name = st.text_input("Actor 2's Name")
        actor_3_name = st.text_input("Actor 3's Name")
    with col3:
        budget = st.number_input("Movie Budget (in Dollars)", min_value=0)
        gross = st.number_input("Movie Gross Earnings (in Dollars)", min_value=0)
        num_voted_users = st.number_input("Number of Voted Users", min_value=0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        num_user_for_reviews = st.number_input("Number of User Reviews", min_value=0)
        num_critic_for_reviews = st.number_input("Number of Critic Reviews", min_value=0)
    with col5:
        title_year = st.number_input("Release Year", min_value=1916, max_value=2016)
        content_rating = st.selectbox("Content Rating", ["G", "PG", "PG-13", "R", "NC-17"])
    with col6:
        color = st.selectbox("Color", ["Color", "Black and White"])
        genres = st.selectbox("Genres", ["Action", "Comedy", "Drama", "Thriller", "Romance", "Horror"])
    
    plot_keywords = st.text_input("Plot Keywords (comma separated)")
    submitted = st.form_submit_button("Classify Movie")

if submitted:
    # Check for empty mandatory fields
    missing_fields = []
    if not movie_title.strip():
        missing_fields.append("Movie Title")
    if duration == 1:
        missing_fields.append("Duration")
    if not director_name.strip():
        missing_fields.append("Director's Name")
    if not actor_1_name.strip():
        missing_fields.append("Actor 1's Name")
    if not actor_2_name.strip():
        missing_fields.append("Actor 2's Name")
    if not actor_3_name.strip():
        missing_fields.append("Actor 3's Name")
    if budget == 0:
        missing_fields.append("Budget")
    if gross == 0:
        missing_fields.append("Gross Earnings")
    if num_voted_users == 0:
        missing_fields.append("Number of Voted Users")
    if num_user_for_reviews == 0:
        missing_fields.append("Number of User Reviews")
    if num_critic_for_reviews == 0:
        missing_fields.append("Number of Critic Reviews")
    if not plot_keywords.strip():
        missing_fields.append("Plot Keywords")

    # Display error message if any field is missing
    if missing_fields:
        st.error(f"The following fields are required and cannot be empty: {', '.join(missing_fields)}")
    else:
        # Prepare the input data
        input_data = pd.DataFrame({
            'movie_title': [movie_title],
            'duration': [duration],
            'director_name': [director_name],
            'actor_1_name': [actor_1_name],
            'actor_2_name': [actor_2_name],
            'actor_3_name': [actor_3_name],
            'budget': [budget],
            'gross': [gross],
            'num_voted_users': [num_voted_users],
            'num_user_for_reviews': [num_user_for_reviews],
            'num_critic_for_reviews': [num_critic_for_reviews],
            'plot_keywords': [plot_keywords],
            'title_year': [title_year],
            'content_rating': [content_rating],
            'color': [color],
            'genres': [genres]
        })

        # Preprocess the data
        preprocessed_data = preprocess_input(input_data[['duration', 'budget', 'gross', 'num_voted_users',
                                                         'num_user_for_reviews', 'num_critic_for_reviews', 'title_year',
                                                         'content_rating', 'color', 'genres']], 
                                             label_encoders, scaler)

        # Predict using the classifier
        prediction = classifier.predict(preprocessed_data)

        # Display the result
        if prediction == 0:
            st.error("Prediction: FLOP Movie")
        elif prediction == 1:
            st.warning("Prediction: AVERAGE Movie")
        else:
            st.success("Prediction: HIT Movie")
