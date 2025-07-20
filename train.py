import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(subset=["ProfileName", "Summary"], inplace=True)
    data = data[data["Score"] != 3]
    data = data[data["HelpfulnessDenominator"] > 0]

    data["sentiment"] = data["Score"].apply(lambda x: 1 if x > 3 else 0)
    data["HelpfulnessRatio"] = data['HelpfulnessNumerator'] / (data['HelpfulnessDenominator'] + 1e-5)

    data["Summary"] = data["Summary"].apply(clean_text)
    data["Text"] = data["Text"].apply(clean_text)
    data["All_Text"] = data["Summary"] + " " + data["Text"]

    return data


def build_pipeline():
    preprocessing = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), ["HelpfulnessRatio"]),
        ("text", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "All_Text")
    ])

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("feature_selector", SelectKBest(score_func=chi2, k=1000)),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            max_depth=None,
            min_samples_split=5,
            n_estimators=200,
            random_state=42
        ))
    ])

    return pipeline


def train(data):
    target = "sentiment"
    features = ["HelpfulnessRatio", "All_Text"]

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(model, "model.pkl")


if __name__ == "__main__":
    file_path = "Reviews.csv"
    data = load_and_preprocess_data(file_path)
    train(data)
