from flask import Flask, jsonify, request
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import os

app = Flask(__name__)

print("Loading model...")  # Debugging step
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
print("Model loaded!")

@app.route('/phonecall', methods=['POST'])
def PhoneCall():
    try:
        url = "https://raw.githubusercontent.com/Mostapha-El-Kaddaoui/Server-Flask-Railway/main/image2.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        Objects = [model.config.id2label[label.item()] for score, label, box in 
                   zip(results["scores"], results["labels"], results["boxes"])]

        return jsonify(Objects)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use Railway's assigned port
    app.run(host='0.0.0.0', port=port, debug=True)
