import numpy as np
import faiss
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import os, pickle

DATASET_FOLDER = 'static/dataset_images'
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = [os.path.join(DATASET_FOLDER, f) for f in os.listdir(DATASET_FOLDER)
                if f.lower().endswith(('jpg', 'jpeg', 'png'))]

embeddings = []
for path in image_paths:
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img_tensor).squeeze().numpy()
    embeddings.append(emb)

embeddings = np.vstack(embeddings).astype('float32')
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

# Save FAISS index and image_paths
faiss.write_index(faiss_index, 'static/index/faiss.index')
with open('static/index/paths.pkl', 'wb') as f:
    pickle.dump(image_paths, f)

print("✅ FAISS index and paths saved.")
