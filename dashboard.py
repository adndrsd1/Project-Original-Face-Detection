import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title="Original Face Detection",
    page_icon=":smiley:",
    initial_sidebar_state='auto'
)

st.header("Original Face Detection")
st.text("This is web app to detect original faces in an image.")

uploaded_file = st.file_uploader("Choose a face image...", type=['jpg'])

if uploaded_file is None:
    st.text("Please upload an image file")
else:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):
        # Load the model
        model = tf.keras.models.load_model("original_face_detection.h5")
        img_height, img_width = 180, 180
        classes = ['fake', 'real']

        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(img_height, img_width))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        image = np.expand_dims(img, axis=0)

        # Make a prediction
        prediction = model.predict(image, batch_size=16)

        # Display the result
        if classes[np.argmax(prediction)] == 'real':
            st.success("This is an original face!")
        else:
            st.error("This is a fake face!")

        st.write(f"Confidence: {prediction}")