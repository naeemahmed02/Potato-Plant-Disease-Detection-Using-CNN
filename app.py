import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

class Detect:

    def __init__(self, model_path, class_names):
        self.model_path = model_path
        self.class_names = class_names
        self.model = self.load_model()

    def load_model(self):
        # Load the pretrained model
        return tf.keras.models.load_model(self.model_path)
    
    # Preprocess the image
    def preprocess_image(self, image):
        img = Image.open(image)
        img = img.resize((256, 256))
        img = np.array(img)
        img = img / 255.0
        return img
    
    # Predict the Image
    def predict(self, image):
        img = self.preprocess_image(image)
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img)
        return predictions

class webapp:
    def __init__(self, classifier):
        self.classifier = classifier

    def run(self):
        st.title("Potato Disease Classifier")

        # Create sidebar menu
        menu_selection = st.sidebar.selectbox("Menu", ["Home", "About"])

        if menu_selection == "Home":
            self.home()
        elif menu_selection == "About":
            self.about()

    def home(self):
        st.write("Upload an image and the model will predict the disease.")

        upload_image = st.sidebar.file_uploader("Choose an image:", type=['jpg','png', 'jpeg'])
        if upload_image is not None:
            image = Image.open(upload_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            predictions = self.classifier.predict(upload_image)
            confidence = np.max(predictions) * 100
            predicted_class = self.classifier.class_names[np.argmax(predictions)]

            if confidence > 80:
                st.write(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}%")
            else:
                st.write("Prediction confidence is below 80%. Please try another image.")

    def about(self):
        st.title("About")
        st.write("This is a Streamlit web app for classifying potato diseases.")

        # Add social media links and icons
        st.markdown(
            """
            <h2>Connect with us:</h2>
            <a href="https://www.linkedin.com/your_page" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png"/>
            </a>
            <a href="https://www.github.com/your_page" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/github--v1.png"/>
            </a>
            <a href="https://www.kaggle.com/your_page" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/kaggle.png"/>
            </a>
            """,
            unsafe_allow_html=True  # Allow HTML rendering
        )

def main():
    model_path = 'plant_village_model11.h5'
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    classifier = Detect(model_path, class_names)
    app = webapp(classifier)
    app.run()

if __name__ == '__main__':
    main()
