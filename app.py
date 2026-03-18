import streamlit as st

from backend.utils import load_data
from backend.client import Client
from backend.federated import federated_training

from PIL import Image
import torch
import numpy as np

st.title("Federated Learning Image Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, width=200)

    loaders = load_data()
    clients = [Client(loader) for loader in loaders]

    model = federated_training(clients)

    img = np.array(image.resize((28,28))) / 255.0
    img_tensor = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)

    output = model(img_tensor)
    prediction = torch.argmax(output, dim=1).item()

    st.success(f"Prediction: {prediction}")