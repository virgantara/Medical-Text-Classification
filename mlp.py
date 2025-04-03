import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test

def train_and_evaluate(X_train, y_train, X_test, y_test, epochs=30):
    clf = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1,           # Only 1 epoch per fit call
        warm_start=True,      # Keep training on same model
        random_state=42,
        verbose=False
    )

    print("Training MLP with progress bar...")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def main():
    train_path = "./data/medical_tc_train.csv"
    test_path = "./data/medical_tc_test.csv"

    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)

    print("Vectorizing text...")
    X_train, X_test = preprocess(train_df['medical_abstract'], test_df['medical_abstract'])

    print("Training MLP and evaluating...")
    epochs = 100
    train_and_evaluate(X_train, train_df['condition_label'], X_test, test_df['condition_label'],epochs)

if __name__ == "__main__":
    main()
