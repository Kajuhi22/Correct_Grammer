from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Load model and tokenizer
model_name = "vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize Flask
app = Flask(__name__)
CORS(app)

# API Endpoint
@app.route("/correct", methods=["POST"])
def correct():
    data = request.get_json()
    input_text = data.get("text", "")
    if not input_text:
        return jsonify({"error": "No text provided"}), 400
    
    inputs = tokenizer.encode("gec: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"corrected": corrected_text})

# Run app on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
