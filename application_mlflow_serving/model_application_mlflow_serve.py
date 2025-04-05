# Import the dependencies we need to run the code.
import os
import gradio as gr
import numpy as np
from keras.layers import TFSMLayer
from keras import Model, Input
import mlflow

# Get a few environment variables. These are so we can:
# - get data from MLFlow
# - Set server name and port for Gradio
MLFLOW_ROUTE = os.getenv("MLFLOW_ROUTE")                    # You need to manually set this with an environment variable
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT"))   # Automatically set by the Dockerfile
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME")        # Automatically set by the Dockerfile

# Connect to MLFlow using the route.
mlflow.set_tracking_uri(MLFLOW_ROUTE)

# Specify what model and version we want to load, and then load it.
model_name = "DNN-credit-card-fraud"
model_version = 1

# Get the path to the model artifact from MLflow
model_uri = f"models:/{model_name}/{model_version}"
local_path = mlflow.artifacts.download_artifacts(model_uri)

# Load the model using TFSMLayer (works with SavedModel in Keras 3)
layer = TFSMLayer(local_path, call_endpoint="serving_default")

# Construct a wrapper Keras model for predict()
inputs = Input(shape=(7,), dtype="float64")  # Assuming 7 features
outputs = layer(inputs)
model = Model(inputs=inputs, outputs=outputs)

# Create a small function that runs predictions on the loaded model.
def predict(distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price,
            repeat_retailer, used_chip, used_pin_number, online_order):
    input_array = np.array([[distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price,
                             repeat_retailer, used_chip, used_pin_number, online_order]], dtype=np.float64)
    prediction = model.predict(input_array)[0][0]
    return "Fraud" if prediction >= 0.995 else "Not fraud"

# Create and launch a Gradio interface
demo = gr.Interface(
    fn=predict, 
    inputs=["number", "number", "number", "number", "number", "number", "number"], 
    outputs="textbox",
    examples=[
        [57.87785658389723, 0.3111400080477545, 1.9459399775518593, 1.0, 1.0, 0.0, 0.0],
        [10.664473716016785, 1.5657690862016613, 4.886520843107555, 1.0, 0.0, 0.0, 1.0]
    ],
    title="Predict Credit Card Fraud"
)

demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
