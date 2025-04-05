import os
import base64
import time
import openai
from fpdf import FPDF
from flask import Flask, request, jsonify, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
from dn import save_as_json,save_as_pdf,process_delivery_note_image
from anpr_app import LicensePlateDetector, TextExtractor
from packmat import ObjectTracker,VideoProcessor
from format_conv import extract_details
'''from extract_number import NumberExtractor

from full_weight_script import process_request
from empty_weight import process_request_empty
from mapping import ask
from ourbd_lpo import process_data'''

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-***************"
client = openai

@app.route("/delivery_note", methods=["POST"])
def process_image_and_generate_delivery_note():
    pdf_folder = os.path.join(os.getcwd(), "pdf_output") 
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
    data = request.get_json()

    if not data or "file_path" not in data:
        return jsonify({"status": "error", "message": "No file path provided"}), 400

    file_path = data["file_path"]

    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File does not exist"}), 400

    if file_path.lower().endswith(".pdf"):
        content = process_delivery_note_image(file_path)
    elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        content = process_delivery_note_image(file_path)
    else:
        return jsonify({"status": "error", "message": "Unsupported file format"}), 400

    if "Error" not in content:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        save_as_json(content, base_filename)
        save_as_pdf(content, base_filename)

        delivery_note_path = os.path.join(pdf_folder, f"{base_filename}_delivery_note.pdf")

        delivery_note_pdf = FPDF()
        delivery_note_pdf.add_page()
        delivery_note_pdf.set_font("Arial", size=12)
        delivery_note_pdf.cell(200, 10, txt="Delivery Note", ln=True)
        delivery_note_pdf.output(delivery_note_path)

        return jsonify({
            "status": "success", 
            "content": extract_details(content), 
            "json": f"{base_filename}.json", 
            "pdf": f"{base_filename}.pdf",
            "delivery_note": delivery_note_path
        }), 200
    else:
        return jsonify({"status": "error", "message": content}), 500

@app.route("/process_anpr_front", methods=["POST"])
def process_anpr():
    data = request.json
    print(data)
    
    image_path = data.get('image_path')
    
    if not data or not image_path:
        return jsonify({"error": "Missing image path"}), 400
    
    try:
        # Initialize paths and model
        model_path = r"license_plate_detector.pt"
        temp_input_folder = r"temp_input"
        temp_output_folder = r"temp_output"
        temp_cropped_folder = r"temp_cropped"
        
        # Initialize detector and text extractor
        license_plate_detector = LicensePlateDetector(model_path, temp_input_folder, temp_output_folder, temp_cropped_folder)
        
        # Process the image to crop and detect the license plate
        cropped_image_path = license_plate_detector.process_image(image_path)
        
        if not cropped_image_path:
            return jsonify({"error": "No license plate detected or cropped."}), 400
        
        license_plate_text = TextExtractor.extract_text(cropped_image_path)
        print(license_plate_text)
        # Read the cropped image and convert it to base64
        with open(cropped_image_path, "rb") as img_file:
            cropped_image_path = os.path.join(temp_cropped_folder, "cropped_image.jpg")
            image_url = url_for('static', filename='temp_cropped/cropped_image.jpg')
        
        return jsonify({
        "license_plate": license_plate_text,  # Replace with your extracted text
        "image_url": image_url  # Return the URL to the cropped image
    }), 200

        # Return both the cropped image and extracted text
        #return jsonify({
            #"license_plate": license_plate_text,
            #"cropped_image": cropped_image_base64
        #}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#anpr process for back-no-plate
@app.route("/process_anpr_back", methods=["POST"])
def process_anpr_back():
    return process_anpr()

#packmat----

@app.route("/process_packmat", methods=["POST"])
def process_video_and_generate_output():
    data = request.get_json()

    if not data or "video_path" not in data:
        return jsonify({"status": "error", "message": "No video path provided"}), 400

    video_path = data["video_path"]

    if not os.path.exists(video_path):
        return jsonify({"status": "error", "message": "File does not exist"}), 400

    output_video_path = os.path.join(os.path.dirname(video_path), "output_video.mp4")

    try:
        processor = VideoProcessor(video_path, model_path=r"jerrycan_bundle_detection.pt", output_path=output_video_path)
        object_count = processor.process_video()

        return jsonify({
            "status": "success",
            "message": "Video processed successfully.",
            "output_video": output_video_path,
            "object_count": object_count
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
