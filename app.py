import pickle

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
