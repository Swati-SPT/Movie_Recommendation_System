# Movie Recommendation System

This project is a Content-Based Movie Recommendation System built using Bag of Words feature extraction and Cosine Similarity.  
It recommends movies similar to a selected movie based on textual features such as genres, keywords, cast, and crew.


## Project Overview

The system analyzes movie metadata and converts relevant text information into numerical vectors using  CountVectorizer.  
Cosine similarity is then used to compute similarity scores between movies and generate recommendations.


## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Natural Language Processing (Bag of Words)  
- Cosine Similarity  


## How It Works

1. Load movie and credit datasets  
2. Merge datasets on movie title  
3. Select important features (genres, keywords, cast, crew, overview)  
4. Preprocess text data  
5. Convert text to vectors using CountVectorizer
6. Compute similarity using Cosine Similarity  
7. Recommend top 5 similar movies  


## How to Run

pip install -r requirements.txt
python recommender.py

