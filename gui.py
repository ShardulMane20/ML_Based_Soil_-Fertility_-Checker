import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

model = load_model('crop_fertilizer_model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('crop_encoder.pkl', 'rb') as file:
    crop_encoder = pickle.load(file)

with open('fertilizer_encoder.pkl', 'rb') as file:
    fertilizer_encoder = pickle.load(file)

def predict():
    try:
        nitrogen = float(nitrogen_entry.get())
        phosphorus = float(phosphorus_entry.get())
        potassium = float(potassium_entry.get())
        ph = float(ph_entry.get())
        rainfall = float(rainfall_entry.get())
        temperature = float(temperature_entry.get())
        
        features = np.array([[nitrogen, phosphorus, potassium, ph, rainfall, temperature]])

        features_scaled = scaler.transform(features)

        crop_pred, fertilizer_pred, fertilizer_usage_pred = model.predict(features_scaled)

        crop_pred = crop_encoder.inverse_transform([np.argmax(crop_pred)])[0]
        fertilizer_pred = fertilizer_encoder.inverse_transform([np.argmax(fertilizer_pred)])[0]
        fertilizer_usage_pred = fertilizer_usage_pred[0][0]

        output_window = tk.Toplevel(window)
        output_window.title("Prediction Results")
        output_window.geometry("600x600")
        output_window.config(bg="#f0f0f0")

        bg_image_path = "D:/Soil_Prediction/Crop_images/bg2.jpg"
        bg_image = Image.open(bg_image_path)
        bg_image = bg_image.resize((600, 600))
        bg_image_tk = ImageTk.PhotoImage(bg_image)
        
        bg_label_output = tk.Label(output_window, image=bg_image_tk)
        bg_label_output.place(relwidth=1, relheight=1)
        bg_label_output.image = bg_image_tk

        title_label = tk.Label(output_window, text="Prediction Results", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        result_label = tk.Label(output_window, text=f"Predicted Crop: {crop_pred}\nPredicted Fertilizer: {fertilizer_pred}\nPredicted Fertilizer Usage: {fertilizer_usage_pred:.2f} kg", font=("Arial", 14), bg="#f0f0f0", justify="left")
        result_label.pack(pady=20)

        display_crop_image(crop_pred, output_window)

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for all features.")

def display_crop_image(crop, output_window):
    crop_images = {
        "Sugarcane": "D:/Soil_Prediction/Crop_images/sugarcane.jpg",
        "Jowar": "D:/Soil_Prediction/Crop_images/jowar.jpg",
        "Cotton": "D:/Soil_Prediction/Crop_images/cotton.jpg",
        "Rice": "D:/Soil_Prediction/Crop_images/rice.jpg",
        "Wheat": "D:/Soil_Prediction/Crop_images/wheat.jpg",
        "Maize": "D:/Soil_Prediction/Crop_images/maize.jpg"
    }

    if crop in crop_images:
        img_path = crop_images[crop]
        try:
            img = Image.open(img_path)
            img = img.resize((400, 400))
            img = ImageTk.PhotoImage(img)
            crop_image_label = tk.Label(output_window, image=img, bg="#f0f0f0")
            crop_image_label.pack(pady=20)
            crop_image_label.image = img
        except Exception as e:
            messagebox.showerror("Image Error", f"Error loading image for {crop}: {e}")
    else:
        messagebox.showwarning("No Image", f"No image available for {crop}")

window = tk.Tk()
window.title("Crop, Fertilizer & Fertilizer Usage Prediction")
window.geometry("700x600")
window.config(bg="#f0f0f0")

bg_image_path = "D:/Soil_Prediction/Crop_images/bg1.jpg"
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((700, 600))
bg_image_tk = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(window, image=bg_image_tk)
bg_label.place(relwidth=1, relheight=1)

title_label_main = tk.Label(window, text="Crop & Fertilizer Prediction Tool", font=("Arial", 20, "bold"), bg="#f0f0f0")
title_label_main.pack(pady=20)

input_frame = tk.Frame(window, bg="#f0f0f0")
input_frame.pack(pady=20)

tk.Label(input_frame, text="Nitrogen (kg/ha):", font=("Arial", 12), bg="#f0f0f0").grid(row=0, column=0, padx=10, pady=5, sticky="w")
nitrogen_entry = tk.Entry(input_frame, font=("Arial", 12))
nitrogen_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Phosphorus (kg/ha):", font=("Arial", 12), bg="#f0f0f0").grid(row=1, column=0, padx=10, pady=5, sticky="w")
phosphorus_entry = tk.Entry(input_frame, font=("Arial", 12))
phosphorus_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Potassium (kg/ha):", font=("Arial", 12), bg="#f0f0f0").grid(row=2, column=0, padx=10, pady=5, sticky="w")
potassium_entry = tk.Entry(input_frame, font=("Arial", 12))
potassium_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(input_frame, text="pH Value:", font=("Arial", 12), bg="#f0f0f0").grid(row=3, column=0, padx=10, pady=5, sticky="w")
ph_entry = tk.Entry(input_frame, font=("Arial", 12))
ph_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Rainfall (mm):", font=("Arial", 12), bg="#f0f0f0").grid(row=4, column=0, padx=10, pady=5, sticky="w")
rainfall_entry = tk.Entry(input_frame, font=("Arial", 12))
rainfall_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Temperature (°C):", font=("Arial", 12), bg="#f0f0f0").grid(row=5, column=0, padx=10, pady=5, sticky="w")
temperature_entry = tk.Entry(input_frame, font=("Arial", 12))
temperature_entry.grid(row=5, column=1, padx=10, pady=5)

predict_button = tk.Button(window, text="Predict", font=("Arial", 14, "bold"), 
                           bg="#4CAF50", fg="white", relief="flat", 
                           width=20, height=2, command=predict)

def on_enter(e):
    predict_button['background'] = '#45a049'

def on_leave(e):
    predict_button['background'] = '#4CAF50'

predict_button.bind("<Enter>", on_enter)
predict_button.bind("<Leave>", on_leave)

predict_button.pack(pady=20)

window.mainloop()
