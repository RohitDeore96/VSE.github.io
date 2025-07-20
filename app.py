import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
import torch

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load FAISS index and image paths
index_path = 'static/index/faiss.index'
paths_path = 'static/index/paths.pkl'

if os.path.exists(index_path) and os.path.exists(paths_path):
    faiss_index = faiss.read_index(index_path)
    with open(paths_path, 'rb') as f:
        image_paths = pickle.load(f)
else:
    faiss_index = None
    image_paths = []
    print("⚠️ Precomputed FAISS index not found.")

# Load pre-trained ResNet18 model with updated weights usage
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC layer
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(img_path):
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor)
    features = features.view(features.size(0), -1)  # Flatten
    features = features.cpu().numpy().astype('float32')
    return features

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part in the request.'

        uploaded_file = request.files['image']
        if uploaded_file.filename == '':
            return 'No file selected.'

        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)

        query_vector = extract_features(file_path)

        if faiss_index is None:
            return "FAISS index not found. Please build the index first."

        if query_vector.shape[1] != faiss_index.d:
            return f"Vector dimension mismatch: expected {faiss_index.d}, got {query_vector.shape[1]}"

        distances, indices = faiss_index.search(query_vector, k=5)

        results = [image_paths[i] for i in indices[0]]

        return render_template('results.html', results=results, query_image='/' + file_path)

    return render_template('index.html')

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    app.run(debug=True)

