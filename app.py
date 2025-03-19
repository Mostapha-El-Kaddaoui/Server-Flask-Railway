from flask import Flask, jsonify, request
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from io import BytesIO
import os

app = Flask(__name__)

# Initialize model and processor outside the route for better performance
processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    return processor, model

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"})

@app.route('/phonecall', methods=['POST'])
def PhoneCall():
    try:
        # Load the model
        processor, model = load_model()
        
        # Get image from URL
        url = "https://raw.githubusercontent.com/Mostapha-El-Kaddaoui/Server-Flask-Railway/refs/heads/main/image2.jpg"
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to fetch image: {response.status_code}'}), 400
            
        image = Image.open(BytesIO(response.content))
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        # Extract detected objects
        Objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            Objects.append(model.config.id2label[label.item()])
        
        # Check the request data
        data = request.json
        if data and data.get('message') == 'hello':
            return jsonify(Objects)
        else:
            return jsonify({'error': 'Invalid request'}), 400
            
    except Exception as e:
        # Handle any errors
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
