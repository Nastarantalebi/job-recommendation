import pandas as pd
import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class ModelTrainClass:   
    def __init__(self, df_embedding):
        self.df = df_embedding
        
    def build_model(self, ClassifierModel_path):
        # Extract features and labels
        X = self.df.iloc[:, :768].values
        y = self.df['main_category_id'].values  

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the SVM model with RBF kernel
        model = SVC(kernel='rbf')

        model.fit(x_train, y_train)

        # Save the trained model
        joblib.dump(model, ClassifierModel_path)

        # Make predictions on the test set
        predictions = model.predict(x_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')


        print("Model built successfuly.")
        n_classes = len(set(y))
        print(f"Number of classes: {n_classes}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Kernel: {model.kernel}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Save evaluation metrics to a text file
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"SVM_evaluation_metrics_{current_date}.txt"
        with open(file_name, "w") as file:
            file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            file.write(f'Precision: {precision * 100:.2f}%\n')
            file.write(f"Recall (Sensitivity): {recall * 100:.2f}%\n")
            file.write(f"F1-Score: {f1 * 100:.2f}%\n")

        print(f"Evaluation Metrics saved to {file_name}")

class ModelInferenceClass:
    def __init__(self, classifier_model_path: str) -> None:
        try:
            self.model = joblib.load(classifier_model_path)
        except FileNotFoundError:
            print(f"Model does not exist in {classifier_model_path}")

    def inference(self, input_embeddings_df):
        if self.model:
            # Extract embeddings
            embeddings = input_embeddings_df.iloc[:, :768].values
            # Make predictions using vectorized operations
            predicted = self.model.predict(embeddings)
            return predicted
        else:
            raise ValueError("Model not loaded.")
        