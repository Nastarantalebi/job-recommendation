import pandas as pd
import datetime
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import tensorflow as tf
from keras.callbacks import History
from keras import utils, Sequential, layers, optimizers, regularizers
from sklearn.metrics import precision_score, recall_score, f1_score


class ModelTrainClass:   
    def __init__(self, df_embedding):
        self.df = pd.DataFrame(df_embedding)

    def build_model(self, ClassifierModel_path):
        # Extract features and labels
        X = self.df.iloc[:, :768].values
        y = self.df['main_category_id'].values  
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # K-Fold Cross Validation
        n_folds = 7
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        
        for train_ix, test_ix in kfold.split(x_train):
            x_train_fold, x_test_fold = x_train[train_ix], x_train[test_ix]
            y_train_fold, y_test_fold = y_train[train_ix], y_train[test_ix]

        #saving a copy of test labels for later useage in calculating metrics
        test_labels_integer_encoded = y_test_fold

        print("y train and test uniqe values:" ,np.unique(y_train_fold) ,np.unique(y_test_fold))

        # converting labels to categorical with one-hot-encoding
        y_train_fold = utils.to_categorical(y_train_fold, num_classes=24)
        y_test_fold = utils.to_categorical(y_test_fold, num_classes=24)

        # Build and compile the model
        model = Sequential()
        model.add(layers.Dense(4096, activation='relu', input_dim=768, kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.4))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(24, activation='softmax'))

        # Compile the model
        opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Initialize History object to store training history
        history = History()

        # Train the model with the History callback
        model.fit(x_train_fold, y_train_fold, validation_data=(x_test_fold, y_test_fold),
            epochs=25, batch_size=5000, callbacks=[history])

        # Evaluate the model
        predictions_probabilities = model.predict(x_test_fold)
        predictions = np.argmax(predictions_probabilities, axis=1)

        # Save the trained model
        model.save(ClassifierModel_path)

        # Calculate evaluation metrics
        _, acc = model.evaluate(x_test_fold, y_test_fold)
        precision = precision_score(test_labels_integer_encoded, predictions, average='weighted')
        recall = recall_score(test_labels_integer_encoded, predictions, average='weighted')
        f1 = f1_score(test_labels_integer_encoded, predictions, average='weighted')

        print("Model built successfuly. Details:")
        n_classes = len(set(y))
        print(f"Model: {model.__class__.__name__}")
        print(f"Number of classes: {n_classes}")
        print(f"Accuracy: {acc * 100:.2f}%")


         # Save evaluation metrics to a text file
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"NN_evaluation_metrics_{current_date}.txt"
        with open(file_name, "w") as file:
            file.write(f'Accuracy: {acc * 100:.2f}%\n')
            file.write(f"Precision: {precision * 100:.2f}%\n")
            file.write(f"Recall (Sensitivity): {recall * 100:.2f}%\n")
            file.write(f"F1-Score: {f1 * 100:.2f}%\n")

        print(f"Evaluation Metrics saved to {file_name}")

        # Generate learning curves
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # Save learning curves to a text file
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"NN_learning_curves_{current_date}.txt"
        with open(file_name, "w") as file:
            file.write("Epochs: " + str(epochs) + "\n")
            file.write("Accuracy: " + str(acc) + "\n")
            file.write("Validation Accuracy: " + str(val_acc) + "\n")
            file.write("Loss: " + str(loss) + "\n")
            file.write("Validation Loss: " + str(val_loss) + "\n")

        print(f"Learning curves saved to {file_name}")

class ModelInferenceClass:
    def __init__(self, classifier_model_path: str) -> None:
        try:
            self.model = tf.keras.models.load_model(classifier_model_path)
        except OSError:
            print(f"Model does not exist in {classifier_model_path}")

    def inference(self, input_embeddings_df):
        if self.model:
            # Extract embeddings and reshape to (n_samples, 768)
            embeddings = input_embeddings_df.iloc[:, :768].values.reshape(-1, 768)
            # Make predictions using vectorized operations
            outputs = self.model(embeddings)
            predicted = tf.argmax(outputs, axis=1).numpy()  # Extract predicted classes as NumPy array
            return predicted
        else:
            raise ValueError("Model not loaded.")