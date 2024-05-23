import streamlit as st
from PIL import Image, ImageDraw
import io
import base64
import os
from roboflow import Roboflow
import tempfile

# Initialize Roboflow client
rf = Roboflow(api_key="vE1rnHyYmah1YSdCVEZY")
project = rf.workspace().project("helmet_detection_ver2")
model = project.version("2").model

def draw_bounding_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    for prediction in predictions:
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        confidence = prediction['confidence']
        class_name = prediction['class']

        # Calculate coordinates
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        # Draw bounding box
        color = "blue"
        if class_name == 'rider':
            color = "yellow"
        elif class_name == 'helmet':
            color = "green"
        elif class_name == 'no-helmet':
            color = "red"

        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        # Draw label
        draw.text((left, top), f"{class_name} {confidence:.2f}", fill="blue")

    return image

def main():
    st.title("Helmet Detection")
    st.markdown("Nguyễn Hải Đăng - 21521920")
    st.markdown("Võ Thái Sơn - 21521833")

    # Interface components
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    uploaded_video = st.file_uploader("Upload Video", type=["mp4"], accept_multiple_files=False)
    confidence_threshold = st.slider("Confidence Threshold", 0, 100, 40)

    if uploaded_image is not None:
        # Display uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_image).convert('RGB')

        # Save image to a temporary file
        temp_file_path = "temp_image.jpg"
        image.save(temp_file_path, format="JPEG")

        # Perform prediction by sending the image to Roboflow's API
        try:
            # Use Roboflow's API to predict
            results = model.predict(temp_file_path, confidence=confidence_threshold, overlap=30).json()
            model.predict(temp_file_path, confidence=confidence_threshold, overlap=30).save("prediction.jpg")
            # Remove temporary file after prediction
            os.remove(temp_file_path)

            # Debug: Print results to inspect the response structure
            st.write("API Response:", results)

            # visualization
            st.image("prediction.jpg", caption="Processed Image", use_column_width=True)

            # Check if the response is successful
            if 'predictions' in results:
                detection_results = results['predictions']
                image_with_boxes = draw_bounding_boxes(image, detection_results)
                st.image(image_with_boxes, caption="Processed Image with Bounding Boxes", use_column_width=True)

                # Count no-helmet detections
                count_no_helmet = sum(1 for detection in detection_results if detection['class'] == 'no-helmet')
                st.warning(f"No-Helmet detected: {count_no_helmet}")
            else:
                st.error("Unexpected response structure from the API")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    if uploaded_video is not None:
        # st.video(uploaded_video)



        try:
            # Save the uploaded video to a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "uploaded_video.mp4")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_video.read())

            # Use Roboflow's API to predict
            job_id, signed_url, expire_time = model.predict_video(
                temp_file_path,
                fps=5,
                prediction_type="batch-video",
            )
            results = model.poll_until_video_results(job_id)


            # Debug: Print results to inspect the response structure
            st.write("API Response:", results)

            # visualization
            


            # Remove temporary file after prediction
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")


if __name__ == "__main__":
    main()
