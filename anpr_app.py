import openai
import os
import shutil
import base64
from ultralytics import YOLO
import streamlit as st


class LicensePlateDetector:
    def __init__(self, model_path, temp_folders):
        self.model = YOLO(model_path)
        self.temp_input_folder = temp_folders["input"]
        self.temp_output_folder = temp_folders["output"]
        self.temp_cropped_folder = temp_folders["cropped"]
        self._initialize_folders()

    def _initialize_folders(self):
        for folder in [self.temp_input_folder, self.temp_output_folder, self.temp_cropped_folder]:
            os.makedirs(folder, exist_ok=True)

    def clear_temp_folders(self):
        for folder in [self.temp_input_folder, self.temp_output_folder, self.temp_cropped_folder]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

    def process_image(self, image_path):
        results = self.model.predict(source=image_path, save=True, save_crop=True)
        cropped_image_path = None

        for result in results:
            result_dir = result.save_dir
            for item in os.listdir(result_dir):
                src_path = os.path.join(result_dir, item)
                if "crops" in item.lower():
                    crop_dir = src_path
                    for sub_folder in os.listdir(crop_dir):
                        sub_folder_path = os.path.join(crop_dir, sub_folder)
                        for crop_item in os.listdir(sub_folder_path):
                            src_crop_path = os.path.join(sub_folder_path, crop_item)
                            cropped_image_path = os.path.join(self.temp_cropped_folder, crop_item)
                            shutil.copy(src_crop_path, cropped_image_path)
        return cropped_image_path


class TextExtractor:
    def __init__(self, api_key):
        openai.api_key = api_key

    def extract_text(self, image_path):
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Corrected from 'client' to 'openai'
        response = openai.ChatCompletion.create(  # Now using openai instead of client
            model='gpt-4o',
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "take out the alphabets and number with special character from the license plate in the image. Use this JSON Schema: " },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
        )
        print(response.choices[0].message.content.strip())
        extracted_text = response.choices[0].message.content.strip()
        return extracted_text


def main():
    st.title("Automatic Number Plate Recognition (ANPR)")

    # Define paths and initialize objects
    model_path = r"license_plate_detector.pt"
    api_key = "sk-***********************"
    temp_folders = {
        "input": "temp_input",
        "output": "temp_output",
        "cropped": "temp_cropped"
    }

    detector = LicensePlateDetector(model_path, temp_folders)
    extractor = TextExtractor(api_key)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        detector.clear_temp_folders()
        input_image_path = os.path.join(detector.temp_input_folder, uploaded_image.name)
        with open(input_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(input_image_path, caption="Uploaded Image", use_column_width=True)
        st.write("Processing the image...")
        cropped_image_path = detector.process_image(input_image_path)

        if cropped_image_path:
            st.image(cropped_image_path, caption="Cropped License Plate", use_column_width=True)
            st.write("Extracting text from the license plate...")
            extracted_text = extractor.extract_text(cropped_image_path)
            if extracted_text:
                st.write(f"**Extracted Text:** {extracted_text}")
            else:
                st.error("Error extracting text from the license plate.")
        else:
            st.error("No license plate detected in the image.")

    if st.button("Clear Temporary Files"):
        detector.clear_temp_folders()
        st.success("Temporary files cleared!")


if __name__ == "__main__":
    main()
