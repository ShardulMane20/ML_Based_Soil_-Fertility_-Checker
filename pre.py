import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import MeanSquaredError
import pickle 

data = pd.read_csv('Crop_and_fertilizer_with_usage.csv')

data = data.drop(columns=['District_Name', 'Link', 'Soil_color'])

X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']]
crop_labels = data['Crop'] 
fertilizer_labels = data['Fertilizer'] 
fertilizer_usage = data['Fertilizer_Usage'] 

crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

y_crop = crop_encoder.fit_transform(crop_labels)
y_fertilizer = fertilizer_encoder.fit_transform(fertilizer_labels)

y_crop_onehot = tf.keras.utils.to_categorical(y_crop, num_classes=len(crop_encoder.classes_))
y_fertilizer_onehot = tf.keras.utils.to_categorical(y_fertilizer, num_classes=len(fertilizer_encoder.classes_))

with open('crop_encoder.pkl', 'wb') as file:
    pickle.dump(crop_encoder, file)

with open('fertilizer_encoder.pkl', 'wb') as file:
    pickle.dump(fertilizer_encoder, file)

for col in X.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

X_train, X_test, y_crop_train, y_crop_test, y_fertilizer_train, y_fertilizer_test, y_usage_train, y_usage_test = train_test_split(
    X_scaled, y_crop_onehot, y_fertilizer_onehot, fertilizer_usage, test_size=0.2, random_state=42
)

input_layer = Input(shape=(X.shape[1],))
x = Dense(128, activation="relu")(input_layer)
x = Dropout(0.2)(x)

crop_output = Dense(len(crop_encoder.classes_), activation="softmax", name="crop_output")(x)

fertilizer_output = Dense(len(fertilizer_encoder.classes_), activation="softmax", name="fertilizer_output")(x)

fertilizer_usage_output = Dense(1, activation="linear", name="fertilizer_usage_output")(x)

model = Model(inputs=input_layer, outputs=[crop_output, fertilizer_output, fertilizer_usage_output])

model.compile(
    optimizer="adam",
    loss={
        "crop_output": CategoricalCrossentropy(),
        "fertilizer_output": CategoricalCrossentropy(),
        "fertilizer_usage_output": "mean_squared_error",
    },
    metrics={
        "crop_output": "accuracy",
        "fertilizer_output": "accuracy",
        "fertilizer_usage_output": MeanSquaredError(),
    }
)

history = model.fit(
    X_train,
    {
        "crop_output": y_crop_train,
        "fertilizer_output": y_fertilizer_train,
        "fertilizer_usage_output": y_usage_train,
    },
    validation_data=(
        X_test,
        {
            "crop_output": y_crop_test,
            "fertilizer_output": y_fertilizer_test,
            "fertilizer_usage_output": y_usage_test,
        },
    ),
    epochs=40,
    batch_size=32
)

model.save("crop_fertilizer_model.h5")
