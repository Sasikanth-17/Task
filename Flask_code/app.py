from flask import Flask, render_template, request, jsonify, Response
import requests

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
HEADERS = {"Authorization": "Bearer API_KEY"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.content

@app.route('/')
def index():
    return render_template('existing_page.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    data = request.get_json()
    user_input = data['input_text']

    # Process the user input here and get the image bytes
    image_bytes = query({
        "inputs": user_input,
    })

    # Create a response with the image bytes
    return Response(response=image_bytes, content_type="image/png")  # Adjust the content type if the image format is different

if __name__ == '__main__':
    app.run(debug=True)
