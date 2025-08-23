# 🖼️ Image Similarity Search using Deep Learning (Flask + ResNet18 + FAISS)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Flask](https://img.shields.io/badge/Flask-Framework-green)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![FAISS](https://img.shields.io/badge/FAISS-SimilaritySearch-orange)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

---

## 📌 Project Overview
This project demonstrates an **Image Similarity Search Engine** built with:
- **Flask (Python web framework)**  
- **ResNet18 (Pre-trained CNN from PyTorch)** for feature extraction  
- **FAISS (Facebook AI Similarity Search)** for fast nearest-neighbor matching  

👉 Upload an image, and the system finds the **most visually similar image** from a dataset.  
This can be used in **e-commerce (product search)**, **digital libraries**, or **image-based recommendation systems**.  

---

## 🎯 Features
- Upload any query image.
- Extracts **deep features (512D embedding)** using ResNet18.
- Uses **FAISS vector search** to find the closest match.
- Returns the **most visually similar image** from the dataset with similarity score.
- Lightweight & simple Flask web app — suitable for **college project demo**.

---

## 🛠️ Tech Stack
- **Frontend:** HTML, CSS (inside Flask templates)  
- **Backend:** Flask (Python)  
- **Deep Learning:** PyTorch (ResNet18)  
- **Search Engine:** FAISS (vector similarity search)  
- **Deployment:** Render / Koyeb / Railway  

---

## 📂 Project Structure

Image-Similarity-App/
│── static/
│ ├── uploads/ # Uploaded query images
│ ├── dataset_images/ # Dataset for searching
│── templates/
│ ├── index.html # Frontend page
│── app.py # Flask backend
│── requirements.txt # Dependencies
│── README.md # Project documentation
