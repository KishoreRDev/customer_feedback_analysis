# Lifecycle stage 5 — Model Building (TF-IDF features)
import pandas as pd
import joblib
 
from sklearn.feature_extraction.text import (TfidfVectorizer)
 
df = pd.read_csv("data/processed/cleaned_reviews.csv")
print(df.shape)
print(df.columns.tolist())
print(df["clean_review"].head(10))
print(df["clean_review"].isna().sum())
print("Dataset shape:", df.shape)

# Handle missing/empty reviews
df["clean_review"] = df["clean_review"].fillna("").astype(str)

df = df[df["clean_review"].str.strip() != ""]

print("After cleaning:", df.shape)
 
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
 
X = vectorizer.fit_transform( df["clean_review"])
 
joblib.dump( vectorizer,"models/tfidf_vectorizer.pkl")
 
print(X.shape)