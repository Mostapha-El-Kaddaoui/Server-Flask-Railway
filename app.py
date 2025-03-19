from flask import Flask, jsonify, request
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import time

app = Flask(__name__)

@app.route('/phonecall', methods=['POST'])
def PhoneCall():
    try:
        # Load the image
        url = "https://github.com/Mostapha-El-Kaddaoui/Server-Flask-Railway/blob/main/compressed_image.jpg"
        image = Image.open(BytesIO(requests.get(url).content))

        # Initialize the processor and model
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

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
            print(Objects)

        # Check the request data
        data = request.json
        if data and data.get('message') == 'hello':
            return jsonify(Objects)  # Use jsonify instead of json.dumps
        else:
            return jsonify({'error': 'Invalid request'}), 400

    except Exception as e:
        # Handle any errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
