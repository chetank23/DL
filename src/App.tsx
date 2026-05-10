import { useState } from "react";
const entries = Array.from({ length: 12 }, (_, i) => `Exp-${i + 1}`);
const fileSizes = [
  "2 KB",
  "5 KB",
  "8 KB",
  "12 KB",
  "16 KB",
  "21 KB",
  "29 KB",
  "34 KB",
  "40 KB",
  "47 KB",
  "53 KB",
  "61 KB",
];
const modifiedTimes = [
  "3 minutes ago",
  "12 minutes ago",
  "28 minutes ago",
  "1 hour ago",
  "2 hours ago",
  "5 hours ago",
  "8 hours ago",
  "yesterday",
  "2 days ago",
  "4 days ago",
  "1 week ago",
  "2 weeks ago",
];
const expShortLabels: Record<string, string> = {
  "Exp-1": "Perceptron Gates",
  "Exp-2": "XOR Neural Net",
  "Exp-3": "MNIST Even/Odd ANN",
  "Exp-4": "Wine Classifier",
  "Exp-5": "IMDB LSTM",
  "Exp-6": "Bike Rental Regression",
  "Exp-7": "Boston Price Predictor",
  "Exp-8": "MNIST CNN",
  "Exp-9": "Diabetes Classifier",
  "Exp-10": "Simple RNN",
  "Exp-11": "Temperature LSTM",
  "Exp-12": "CIFAR-10 CNN",
};
const exp1Code = `import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)
        self.lr = learning_rate
    def activation(self, x):
        return 1 if x >= 0 else 0
    def predict(self, x):
        x = np.insert(x, 0, 1)
        return self.activation(np.dot(self.weights, x))
    def train(self, training_inputs, labels, epochs=10):
        for _ in range(epochs):
            for x, y in zip(training_inputs, labels):
                x = np.insert(x, 0, 1)
                prediction = self.activation(np.dot(self.weights, x))
                self.weights += self.lr * (y - prediction) * x
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
and_labels = np.array([0,0,0,1])
and_gate = Perceptron(2)
and_gate.train(inputs, and_labels)
print("AND Gate")
for i in inputs:
    print(i, "->", and_gate.predict(i))
or_labels = np.array([0,1,1,1])
or_gate = Perceptron(2)
or_gate.train(inputs, or_labels)
print("\\nOR Gate")
for i in inputs:
    print(i, "->", or_gate.predict(i))
`;
const exp2Code = `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))
    def forward(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output
    def backward(self, inputs, expected_output, actual_output):
        output_error = expected_output - actual_output
        d_output = output_error * sigmoid_derivative(actual_output)
        hidden_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_error * sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += inputs.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate
    def train(self, inputs, outputs, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(inputs)
            self.backward(inputs, outputs, output)
            if epoch % 1000 == 0:
                loss = np.mean((outputs - output) ** 2)
                print(f"Epoch {epoch} Loss : {loss:.4f}")
    def predict(self, inputs):
        return np.round(self.forward(inputs))
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])
nn = NeuralNetwork(2, 2, 1)
nn.train(X, y)
print("\\nPredicted Outputs")
for x in X:
    prediction = int(nn.predict(np.array([x]))[0][0])
    print(f"Input : {x} -> Predicted : {prediction}")
`;
const exp3Code = `import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = (y_train % 2).astype("int32")
y_test = (y_test % 2).astype("int32")
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)
loss, accuracy = model.evaluate(x_test, y_test)
print("\\nFinal Results")
print(f"Loss : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f}")
`;
const exp4Code = `import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
columns = [
    'Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash',
    'Magnesium', 'Total Phenols', 'Flavanoids',
    'Nonflavanoid Phenols', 'Proanthocyanins',
    'Color Intensity', 'Hue', 'OD280/OD315', 'Proline'
]
wine_data = pd.read_csv(url, header=None, names=columns)
X = wine_data.iloc[:, 1:].values
y = wine_data.iloc[:, 0].values - 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=5,
    validation_data=(X_test, y_test)
)
test_loss, test_acc = model.evaluate(X_test, y_test)
predictions = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, predictions)
print("\\nFinal Results")
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Prediction Accuracy : {acc:.4f}")
`;
const exp5Code = `import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
max_len = 500
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
model = Sequential()
model.add(layers.Embedding(input_dim=max_words, output_dim=128))
model.add(layers.LSTM(128))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\\nFinal Results")
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
predictions = model.predict(x_test[:5])
print("\\nPredictions")
for i, pred in enumerate(predictions):
    print(f"Review {i+1} Prediction : {pred[0]:.4f}")
print("\\nEnter a review using word indices separated by space")
print("Example : 1 14 20 43 5")
user_input = input("Enter Review : ")
user_review = [int(i) for i in user_input.split()]
user_review = pad_sequences([user_review], maxlen=max_len)
prediction = model.predict(user_review)[0][0]
if prediction >= 0.5:
    print("\\nPredicted Sentiment : POSITIVE")
else:
    print("\\nPredicted Sentiment : NEGATIVE")
`;
const exp6Code = `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

day_data = pd.read_csv("bikerent_data.csv")
day_data = day_data.drop(columns=["instant", "dteday", "casual", "registered"])
X = day_data.drop("cnt", axis=1)
y = day_data["cnt"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
early_stop = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)
loss, mae = model.evaluate(X_test, y_test)
print("\\nFinal Results")
print(f"Test MAE : {mae:.2f}")
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE : {mse:.2f}")
print(f"R2 Score : {r2:.2f}")
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Training Performance")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
print("\\nEnter details to predict bike rentals")
user_input = {}
for col in X.columns:
    val = float(input(f"Enter value for {col}: "))
    user_input[col] = val
user_df = pd.DataFrame([user_input])
user_scaled = scaler.transform(user_df)
prediction = model.predict(user_scaled)[0][0]
print(f"\\nPredicted Bike Rentals : {int(prediction)}")
`;
const exp7Code = `import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Dense

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)
X = data.drop('medv', axis=1)
y = data['medv']
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], 1, X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = keras.Sequential()
model.add(
    SimpleRNN(
        128,
        activation='relu',
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)
model.add(Dense(1))
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32
)
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error: {mae}")
`;
const exp8Code = `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train_eo = (y_train % 2).astype("int32")
y_test_eo = (y_test % 2).astype("int32")
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    x_train,
    y_train_eo,
    epochs=5,
    validation_data=(x_test, y_test_eo)
)
loss, accuracy = model.evaluate(
    x_test,
    y_test_eo,
    verbose=0
)
print(f"Test Loss : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f}")
`;
const exp9Code = `import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]
df = pd.read_csv(
    url,
    header=None,
    names=column_names
)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential([
    Dense(16, input_shape=(8,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10,
    verbose=1
)
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
    
print(f"Test Accuracy : {accuracy:.2f}")
`;
const exp10Code = `import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split

np.random.seed(42)
timesteps = 10
features = 1
samples = 1000
X = np.random.rand(samples, timesteps, features)
y = np.random.randint(0, 2, samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = keras.Sequential()
model.add(SimpleRNN(32, activation='relu', input_shape=(timesteps, features)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")
`;
const exp11Code = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
data = pd.read_csv(url)
values = data['Temp'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)
def create_sequences(data, time_steps=10):
    X = []
    y = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)
time_steps = 10
X, y = create_sequences(scaled_data, time_steps)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, 1)),
    Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32
)
loss = model.evaluate(X_test, y_test)
print(f"\\nTest MSE : {loss:.4f}")
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)
plt.plot(y_test_actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()
print("\\nPredict Next Day Temperature")
print("\\nEnter last 10 temperature values separated by space")
user_input = input(">> ")
user_values = list(map(float, user_input.split()))
if len(user_values) != 10:
    print("Please enter exactly 10 values")
else:
    user_array = np.array(user_values).reshape(-1, 1)
    user_scaled = scaler.transform(user_array)
    user_scaled = user_scaled.reshape(1, 10, 1)
    predicted_scaled = model.predict(user_scaled)
    predicted_temp = scaler.inverse_transform(predicted_scaled)
    print(f"Predicted Next Day Temperature : {predicted_temp[0][0]:.2f} °C")
`;
const exp12Code = `import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
datagen = ImageDataGenerator(horizontal_flip=True)
datagen.fit(x_train)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=x_train.shape[1:]),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=5, validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\\nTest Loss : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
`;
const expFiles: Record<
  string,
  { title: string; filename: string; code: string }
> = {
  "Exp-1": {
    title: "Perceptron for AND and OR Gates",
    filename: "perceptron_gate.py",
    code: exp1Code,
  },
  "Exp-2": {
    title: "XOR using a Two-Layer Neural Network",
    filename: "xor_neural_network.py",
    code: exp2Code,
  },
  "Exp-3": {
    title: "MNIST Even/Odd Digit Classification (ANN)",
    filename: "mnist_even_odd_classifier.py",
    code: exp3Code,
  },
  "Exp-4": {
    title: "Wine Dataset Multiclass Classification",
    filename: "wine_classification_nn.py",
    code: exp4Code,
  },
  "Exp-5": {
    title: "IMDB Sentiment Analysis using LSTM",
    filename: "imdb_lstm_sentiment.py",
    code: exp5Code,
  },
  "Exp-6": {
    title: "Bike Rental Demand Regression",
    filename: "bike_rental_regression.py",
    code: exp6Code,
  },
  "Exp-7": {
    title: "Boston Housing Price Prediction using RNN",
    filename: "boston_rnn_regression.py",
    code: exp7Code,
  },
  "Exp-8": {
    title: "MNIST Even/Odd Classification using CNN",
    filename: "mnist_cnn_even_odd.py",
    code: exp8Code,
  },
  "Exp-9": {
    title: "PIMA Diabetes Prediction (Binary Classification)",
    filename: "pima_diabetes_classifier.py",
    code: exp9Code,
  },
  "Exp-10": {
    title: "Simple RNN Binary Classifier",
    filename: "simple_rnn_binary_classifier.py",
    code: exp10Code,
  },
  "Exp-11": {
    title: "Time Series Temperature Forecasting with LSTM",
    filename: "temperature_lstm_forecast.py",
    code: exp11Code,
  },
  "Exp-12": {
    title: "CIFAR-10 Image Classification using CNN",
    filename: "cifar10_cnn_classifier.py",
    code: exp12Code,
  },
};
function App() {
  const [activeExp, setActiveExp] = useState<string | null>(null);
  const [copyLabel, setCopyLabel] = useState("Copy Code");
  const activeFile = activeExp ? expFiles[activeExp] : null;
  const handleCopy = async () => {
    if (!activeFile) return;
    await navigator.clipboard.writeText(activeFile.code);
    setCopyLabel("Copied!");
    window.setTimeout(() => setCopyLabel("Copy Code"), 1200);
  };
  return (
    <div className="jupyter-shell">
      <header className="top-header">
        <div className="brand">
          <img className="brand-logo" src="/jupyter-logo.png" alt="jupyter" />
        </div>
      </header>
      <div className="menu-strip">
        <button className="menu-item" type="button">
          File
        </button>
        <button className="menu-item" type="button">
          View
        </button>
        <button className="menu-item" type="button">
          Settings
        </button>
        <button className="menu-item" type="button">
          Help
        </button>
      </div>
      {activeFile ? (
        <main className="content exp-content">
          <section className="exp-page" aria-label={`${activeExp} detail page`}>
            <div className="exp-page-top">
              <button
                className="small-btn"
                type="button"
                onClick={() => setActiveExp(null)}
              >
                ← Back to Files
              </button>
              <button className="copy-btn" type="button" onClick={handleCopy}>
                {copyLabel}
              </button>
            </div>
            <h2 className="exp-title">
              {activeExp}: {activeFile.title}
            </h2>
            <p className="exp-file-name">{activeFile.filename}</p>
            <pre className="code-block">{activeFile.code}</pre>
          </section>
        </main>
      ) : (
        <main className="content">
          <section className="main-panel" aria-label="Notebook files">
            <div className="panel-top">
              <nav className="tabs" aria-label="Sections">
                <button className="tab active" type="button">
                  Files
                </button>
                <button className="tab" type="button">
                  Running
                </button>
                <button className="tab" type="button">
                  Clusters
                </button>
              </nav>
              <div className="table-toolbar">
                <button className="small-btn" type="button">
                  New <span aria-hidden="true">▼</span>
                </button>
                <button className="small-btn" type="button">
                  Upload
                </button>
                <button className="icon-btn" type="button" aria-label="Refresh">
                  ↻
                </button>
              </div>
            </div>
            <p className="helper-text">
              Select items to perform actions on them.
            </p>
            <section className="table-wrap">
              <div className="list-header">
                <div className="list-header-left">
                  <input type="checkbox" aria-label="Select header row" />
                </div>
                <div className="list-header-right">
                  <span className="list-col-name">
                    Name <span aria-hidden="true">▲</span>
                  </span>
                  <span className="list-col-date">Last Modified</span>
                  <span className="list-col-size">File Size</span>
                </div>
              </div>
              <table className="file-table">
                <tbody>
                  {entries.map((name, index) => (
                    <tr key={name}>
                      <td className="check-col">
                        <input type="checkbox" aria-label={`Select ${name}`} />
                      </td>
                      <td className="name-cell">
                        <div className="name-content">
                          <button
                            className="name-link"
                            type="button"
                            onClick={() => setActiveExp(name)}
                          >
                            <span className="exp-number">{name}</span>
                            <span className="exp-sep"> - </span>
                            <span className="exp-short-name">
                              {expShortLabels[name]}
                            </span>
                          </button>
                        </div>
                      </td>
                      <td className="date-col">{modifiedTimes[index]}</td>
                      <td className="size-col">{fileSizes[index]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          </section>
        </main>
      )}
    </div>
  );
}
export default App;
