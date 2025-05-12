import streamlit as st
import numpy as np
import tensorflow as tf
import base64
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import models
from PIL import Image

# Load the trained model
MODEL_PATH = "models/trained.h5"
model = load_model(MODEL_PATH,compile=False)

# Class Labels
classes = ['Normal', 'Pneumonia']

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((255, 255))
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image



def generate_gradcam(image_tensor, model, layer_name="block5_conv3"):
    grad_model = Model(
        inputs=[model.input],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize heatmap using PIL
    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize((255, 255), resample=Image.BILINEAR)
    heatmap = np.array(heatmap)

    # Apply colormap using matplotlib
    import matplotlib.cm as cm
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap / 255.0)  # returns RGBA
    heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])  # Drop alpha

    return heatmap_colored


def download_report(pred_class, confidence):
    report_text = f""" 
    Chest X-Ray Classification Report
    ----------------------------------
    Prediction : {pred_class}
    Confidence : {confidence:.2f}%
    """
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">Download Report</a>'
    return href

# Streamlit UI
st.set_page_config(page_title="Pneumonia Detection", layout="wide")
st.sidebar.title("Upload Chest X-Ray Picture")
uploaded_file = st.sidebar.file_uploader("Select an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-Ray")
    
    if st.button("Classify Image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        pred_idx = np.argmax(output, axis=1)[0]
        confidence = output[0][pred_idx] * 100
        pred_class = classes[pred_idx]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")
        with col2:
            gradcam_image = generate_gradcam(input_tensor, model)
            st.image(gradcam_image, caption="Grad-CAM Visualization")
        
        st.markdown(download_report(pred_class, confidence), unsafe_allow_html=True)

