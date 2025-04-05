import os
import base64
import json
import openai
from fpdf import FPDF
from pdf2image import convert_from_path

os.environ["OPENAI_API_KEY"] ="sk-***********************"

class DeliveryNoteTextExtractor:
    def __init__(self, api_key):
        api_key = "sk-***********************"
        openai.api_key = api_key

    def extract_text(self, image_path):
        try:
            if image_path.lower().endswith(".pdf"):
                image_path = self.convert_pdf_to_image(image_path)

            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

            response = openai.ChatCompletion.create(
                model='gpt-4o',
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the following details from the delivery note:"},
                            {"type": "text", "text": "Delivery Note Number"},
                            {"type": "text", "text": "LPO Number"},
                            {"type": "text", "text": "Driver Name"},
                            {"type": "text", "text": "Driver Contact"},
                            {"type": "text", "text": "Truck Number"},
                            {"type": "text", "text": "Tank Number"},
                            {"type": "text", "text": "Point of Origin"},
                            {"type": "text", "text": "Final Destination"},
                            {"type": "text", "text": "Port of Discharge"},
                            {"type": "text", "text": "Country of Final Destination"},
                            {"type": "text", "text": "1st Weight"},
                            {"type": "text", "text": "2nd Weight"},
                            {"type": "text", "text": "Net Weight"},
                            {"type": "text", "text": "Products(Name, Quantity, Unit Price, Total Price, Batch Number, LPO Number)"},
                            {"type": "text", "text": "OCR Seals"},
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

            if not response or not response.choices or not response.choices[0].message.content:
                return "Error: No content returned"

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"

    def convert_pdf_to_image(self, pdf_path):
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        image_path = "temp_page.jpg"
        images[0].save(image_path, "JPEG")
        return image_path

def save_as_json(content, filename):
    json_filename = f"{filename}.json"
    parsed_content = {"content": content}
    with open(json_filename, 'w') as json_file:
        json.dump(parsed_content, json_file, indent=4)

def save_as_pdf(content, filename):
    pdf_filename = f"{filename}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    lines = content.split("\n")
    for line in lines:
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(pdf_filename)

def process_delivery_note_image(file_path):
    extractor = DeliveryNoteTextExtractor(api_key=os.environ["OPENAI_API_KEY"])
    content = extractor.extract_text(file_path)

    if "Error" not in content:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        save_as_json(content, base_filename)
        save_as_pdf(content, base_filename)
        return content
    else:
        return None

a = process_delivery_note_image(r"delivernote_sample.pdf")
print(a)
