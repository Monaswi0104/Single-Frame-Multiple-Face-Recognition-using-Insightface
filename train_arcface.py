import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from insightface.app import FaceAnalysis
from tqdm import tqdm

# Initialize InsightFace with ArcFace (antelopev2)
# antelopev2 includes:
#   - scrfd_10g_bnkps.onnx  → SCRFD face detector (fast multi-face detection)
#   - glintr100.onnx        → ArcFace GlintR100 recognition (ResNet-100, trained on Glint360K)
#   - 1k3d68.onnx           → 3D face landmark model
#   - 2d106det.onnx         → 2D landmark model
#   - genderage.onnx        → Gender/age estimation
print("Loading InsightFace ArcFace model (antelopev2)...")
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded successfully!")

# Path to dataset
DATASET_PATH = "/Users/monaswi/Documents/mini-project/augmented-dataset"
OUTPUT_FILE = "face_embeddings_arcface.pkl"

# Dictionary to store embeddings
face_db = {}
embedding_vectors = []
labels = []

# Process each person's folder
print(f"\nProcessing dataset from {DATASET_PATH}...")

person_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

for person_name in tqdm(person_folders, desc="Processing Classes"):
    person_path = os.path.join(DATASET_PATH, person_name)
    embeddings = []

    image_files = [f for f in os.listdir(person_path) if not f.startswith('.')]
    for image_name in tqdm(image_files, desc=f"  Images ({person_name})", leave=False):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        # InsightFace expects BGR input (OpenCV default), so no conversion needed
        faces = app.get(img)
        if len(faces) > 0:
            # normed_embedding is already L2-normalized by InsightFace
            embeddings.append(faces[0].normed_embedding)

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        # Re-normalize the averaged embedding
        avg_embedding /= np.linalg.norm(avg_embedding)
        face_db[person_name] = avg_embedding

        embedding_vectors.append(avg_embedding)
        labels.append(person_name)

# Save embeddings to file
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(face_db, f)

print(f"\nTraining complete! Embeddings saved to {OUTPUT_FILE}")

# Convert embeddings to numpy array
embedding_vectors = np.array(embedding_vectors)
print(f"Total classes recorded: {len(embedding_vectors)}")

# Perform t-SNE for better visualization
if len(embedding_vectors) > 1:
    perplexity = min(5, len(embedding_vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=50, random_state=42)
    reduced_embeddings = tsne.fit_transform(embedding_vectors)

    # Plot face clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='deep', s=100)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("ArcFace Embeddings Clustering using t-SNE")
    plt.legend()
    plt.savefig('arcface_tsne.png')
    plt.show()
else:
    print("Not enough varied embeddings to plot t-SNE.")
