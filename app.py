from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline
from PIL import Image
import os
import torch

# Initialize Flask app
app = Flask(__name__, static_folder="static")

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


# Load models
from transformers import pipeline ,AutoModelForCausalLM ,AutoModelForSequenceClassification, AutoTokenizer

model_path = "./fine_tuned_bart_mnli_fever"

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a text-generation pipeline
text_classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
# Load fine-tuned model
from transformers import ViTForImageClassification, ViTImageProcessor
# Path to the fine-tuned model
save_path = "./fine_tuned_vit_deepfake"
model = ViTForImageClassification.from_pretrained(save_path)
processor = ViTImageProcessor.from_pretrained(save_path)

# Create the image classification pipeline
image_classifier = pipeline("image-classification", model=model, feature_extractor=processor)

# Serve the index.html
from flask import Flask, render_template
#@app.route("/")
#def home():
#    return send_from_directory(app.static_folder, "index.html")
@app.route('/')
def index():
    return render_template('index.html')

#Route for text classification
@app.route("/check_text", methods=["POST"])
def check_text():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = text_classifier(text, candidate_labels=["True", "False", "Unverified"])
    return jsonify({"result": result["labels"][0]})

# Route for image classification
# Route for image classification
@app.route("/check_image", methods=["POST"])
def check_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(file.stream)
    
    # Classify the image
    result = image_classifier(image)

    # The result is usually a list with one dictionary, containing the label and score.
    # Example: [{'label': 'LABEL_NAME', 'score': 0.99}]
    top_label = result[0]['label']
    score = result[0]['score']

    # Map the result based on the labels from your dataset
    if top_label == "LABEL_0":  # 'LABEL_0' corresponds to Fake
        classification = "Fake"
    elif top_label == "LABEL_1":  # 'LABEL_1' corresponds to Real
        classification = "Real"
    else:
        classification = "Unknown"

    return jsonify({"result": classification, "score": score})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
