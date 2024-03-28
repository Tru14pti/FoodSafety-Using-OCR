from flask import Flask, render_template, request, jsonify,redirect, url_for, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import time
import requests
import os
from werkzeug.utils import secure_filename
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Safety Scan AI"
# Define conversation state variables
conversation_state = {"step": 0, "product_name": None}


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

UPLOAD_FOLDER = "F:/pytorch-chatbot-master/uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    return render_template('chat.html')

 
@app.route('/uploads/<name>')
def display_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route("/get", methods=["GET", "POST"])
def chat():
    global conversation_state
    msg = request.form["msg"]
    input = msg
    
    
    
    # Handle conversation flow
    if conversation_state["step"] == 0:
        conversation_state["step"] = 1
        return "Hi! How can I help you?"
    elif conversation_state["step"] == 1:
        # Assuming user's response is an image upload
        conversation_state["step"] = 2
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)
    
        return "Please upload the photo of ingredients."
    elif conversation_state["step"] == 2:
        # Assuming user's response is the product name
        conversation_state["product_name"] = input
        conversation_state["step"] = 3
        return "Specify the product name."
    elif conversation_state["step"] == 3:
        conversation_state["step"] = 4 
        return "In what format do you want the result? Click on the above button."
    elif conversation_state["step"] == 4:
        conversation_state["step"] = 5
        return "Thank you for visiting."
    elif conversation_state["step"] == 5:
        # Reset conversation state after completion
        conversation_state = {"step": 0, "product_name": None}
        return "Hi! How can I help you?"  # Restart the conversation
    
    return get_Chat_response(input)

bot_name = "Safety Scan AI"

def get_Chat_response(text):
    while True:
        sentence = text
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return " I do not understand..."
    # sentence = "do you use credit cards?"
     

if __name__ == '__main__':
    app.run()
