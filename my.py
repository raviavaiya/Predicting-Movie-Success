import streamlit as st
import pandas as pd
import numpy as np
# import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def preprocess_input(input_data, label_encoders, scaler):
    # Apply label encoding for categorical columns
    # input_data['director_name'] = label_encoders['director_name'].transform(input_data['director_name'])
    input_data['genres'] = label_encoders['genres'].transform(input_data['genres'])
    # input_data['actor_1_name'] = label_encoders['actor_1_name'].transform(input_data['actor_1_name'])

    # Select numerical columns and scale them
    numerical_data = input_data[['duration', 'gross', 'num_voted_users', 
                                 'num_user_for_reviews', 'budget', 'title_year','movie_facebook_likes']]
    scaled_data = scaler.transform(numerical_data)
    
    # Add encoded categorical features back to the scaled data
    categorical_data = input_data[['genres']].values
    final_data = np.hstack((scaled_data, categorical_data))
    
    return final_data


def train_model():
    # Load your actual dataset from a CSV file
    data = pd.read_csv("Dataset/movie_metadata.csv")  # Replace with the path to your CSV file
    print('Dataset loaded successfully')

    # Categorizing the target variable based on the 'imdb_score'
    bins = [1, 3, 6, 10]
    labels = ['FLOP', 'AVG', 'HIT']
    data['imdb_success'] = pd.cut(data['imdb_score'], bins=bins, labels=labels)

    # Drop rows with missing values (NaN)
    data.dropna(inplace=True)

   #Removing few columns due to multicollinearity
    data.drop(columns=['director_facebook_likes','actor_3_facebook_likes','actor_2_name','actor_3_name','actor_2_facebook_likes','aspect_ratio','actor_1_facebook_likes','facenumber_in_poster','plot_keywords'],inplace=True)
    data.drop(columns=['cast_total_facebook_likes','num_critic_for_reviews','color','language'],inplace=True)
    data.drop(columns=['country','content_rating','director_name','actor_1_name'],inplace=True)

    # Assuming that 'imdb_success' is the target variable
    X_train = data[['duration', 'gross', 'genres',
       'num_voted_users', 'num_user_for_reviews', 'budget', 'title_year',
       'movie_facebook_likes']]

    y_train = data['imdb_success']  # This is the target variable

    # Label Encoding categorical columns
    label_encoders = {
        'genres': LabelEncoder()
    }
    for col in label_encoders.keys():
        X_train[col] = label_encoders[col].fit_transform(X_train[col])

    # Scaling numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[['duration', 'gross', 'genres',
       'num_voted_users', 'num_user_for_reviews', 'budget', 'title_year',
       'movie_facebook_likes']])

    # Add categorical features back after encoding
    X_train_scaled = np.hstack((X_train_scaled, X_train[['genres']].values))

    # Create a RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

    # Train the model
    classifier.fit(X_train_scaled, y_train)

    return classifier, label_encoders, scaler


# classifier=joblib.load('Training/random_forest_model.joblib')

# label_encoders = {
#         'genres': LabelEncoder()
#     }

# scaler = StandardScaler()

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
  
    with col3:
        budget = st.number_input("Movie Budget (in Dollars)", min_value=0)
        gross = st.number_input("Movie Gross Earnings (in Dollars)", min_value=0)
        num_voted_users = st.number_input("Number of Voted Users", min_value=0)
        movie_facebook_likes = st.number_input("Movie Facebook Likes", min_value=0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        num_user_for_reviews = st.number_input("Number of User Reviews", min_value=0)
    with col5:
        title_year = st.number_input("Release Year", min_value=1916, max_value=2016)
    with col6:
        genres = st.selectbox("Genres", ["Action", "Comedy", "Drama", "Thriller", "Romance", "Horror"])
    
    submitted = st.form_submit_button("Classify Movie")

if submitted:
    # Check for empty mandatory fields
    missing_fields = []
    if not movie_title.strip():
        missing_fields.append("Movie Title")
    if duration == 1:
        missing_fields.append("Duration")
   
    if budget == 0:
        missing_fields.append("Budget")
    if gross == 0:
        missing_fields.append("Gross Earnings")
    if num_voted_users == 0:
        missing_fields.append("Number of Voted Users")
    if num_user_for_reviews == 0:
        missing_fields.append("Number of User Reviews")
    if movie_facebook_likes == 0:
        missing_fields.append("Movie Facebook Likes")
    
    # Display error message if any field is missing
    if missing_fields:
        st.error(f"The following fields are required and cannot be empty: {', \n'.join(missing_fields)}")
    else:
        # Prepare the input data
        input_data = pd.DataFrame({
            'duration': [duration],
            'gross': [gross],
            'genres': [genres],
            'num_voted_users': [num_voted_users],
            'num_user_for_reviews': [num_user_for_reviews],
            'budget': [budget],
            'title_year': [title_year],
            'movie_facebook_likes': [movie_facebook_likes],
            'movie_title': [movie_title]
        })

        # Preprocess the data
        preprocessed_data = preprocess_input(input_data[[ 'duration', 'gross', 'genres', 
       'num_voted_users', 'num_user_for_reviews', 'budget', 'title_year',
       'movie_facebook_likes']], label_encoders, scaler)

        # Predict using the classifier
        prediction = classifier.predict(preprocessed_data)

        # Display the result
        if prediction == 0:
            st.error("Prediction: FLOP Movie")
        elif prediction == 1:
            st.warning("Prediction: AVERAGE Movie")
        else:
            st.success("Prediction: HIT Movie")