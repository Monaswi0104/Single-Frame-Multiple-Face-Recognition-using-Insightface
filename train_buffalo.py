import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
from insightface.app import FaceAnalysis
from tqdm import tqdm
import pickle

# Initialize Face Recognition Model  
# Using InsightFace's ArcFace model ("buffalo_l") with CPU processing.  
# The model is prepared to detect and extract facial embeddings from images.  
app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Define Dataset Path and Output File  
# The dataset contains multiple folders, each representing a different person.  
# Extracted face embeddings will be saved to a file for future use.  
DATASET_PATH = "augmented-dataset"
OUTPUT_FILE = "face_embeddings.pkl"

# Initialize Embeddings and Labels Storage  
# This list stores feature vectors representing facial embeddings.  
# Labels correspond to the folder names (individual identities).  
embedding_vectors = []
labels = []

# Apply Data Augmentation Techniques  
# This pipeline introduces variations such as horizontal flips,  
# small rotations, and brightness adjustments to improve robustness.  
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-10, 10)),
    iaa.GammaContrast((0.8, 1.2))
])

# Process Images in Each Folder  
# Each folder represents a different person. For each image,  
# facial embeddings are extracted and augmented variations are also processed.  
people = [p for p in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, p))]
for person_name in tqdm(people, desc="Processing People"):
    person_path = os.path.join(DATASET_PATH, person_name)
    
    images = [img for img in os.listdir(person_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
    for image_name in tqdm(images, desc=f"Scanning {person_name}", leave=False):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract facial embeddings using the ArcFace model  
        faces = app.get(img)
        if len(faces) > 0:
            embedding_vectors.append(faces[0].normed_embedding)
            labels.append(person_name)

        # Generate and process augmented versions of the image  
        aug_img = augmenter.augment_image(img)
        faces_aug = app.get(aug_img)
        if len(faces_aug) > 0:
            embedding_vectors.append(faces_aug[0].normed_embedding)
            labels.append(person_name)

# --- Save Embeddings and Generate Similarity Distribution ---
raw_embeddings = np.array(embedding_vectors)
unique_labels = np.unique(labels)
face_db = {}

# Compute mean embedding for each person
for name in unique_labels:
    person_embs = raw_embeddings[np.array(labels) == name]
    mean_emb = np.mean(person_embs, axis=0)
    face_db[name] = mean_emb / np.linalg.norm(mean_emb)

# Save embeddings for the recognition script
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(face_db, f)

# --- 1. Dataset Distribution Graph ---
plt.figure(figsize=(12, 6))
sns.countplot(x=labels, hue=labels, palette='viridis', order=np.unique(labels), legend=False)
plt.title("Dataset Distribution: Images per Person")
plt.xlabel("Person")
plt.ylabel("Number of Augmented Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dataset_distribution.png", dpi=300)
plt.show()

# --- 2. Intra-class vs Inter-class Similarity & ROC Curve ---
print("Computing pairwise similarities for ROC and Distribution graphs...")
normed_embs = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
sim_matrix = np.dot(normed_embs, normed_embs.T)

intra_sims = []
inter_sims = []
y_true = []
y_scores = []

# Get upper triangle indices to avoid duplicate pairs and self-pairs
i_indices, j_indices = np.triu_indices(len(labels), k=1)

for i, j in zip(i_indices, j_indices):
    sim = sim_matrix[i, j]
    is_same = (labels[i] == labels[j])
    
    y_true.append(1 if is_same else 0)
    y_scores.append(sim)
    
    if is_same:
        intra_sims.append(sim)
    else:
        inter_sims.append(sim)

# Plot Similarity Distributions (Intra vs Inter)
plt.figure(figsize=(10, 6))
sns.histplot(inter_sims, bins=50, kde=True, color='skyblue', label='Inter-class (Different Person)', stat='density', alpha=0.6)
sns.histplot(intra_sims, bins=50, kde=True, color='lightgreen', label='Intra-class (Same Person)', stat='density', alpha=0.6)
plt.title("Intra-class vs Inter-class Cosine Similarity")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("similarity_distribution.png", dpi=300)
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# --- Normalize and Transform Embeddings for t-SNE ---  
# Convert embeddings into a NumPy array and standardize them  
# to improve the performance of dimensionality reduction algorithms.  
embedding_vectors = np.array(embedding_vectors)
embedding_vectors = StandardScaler().fit_transform(embedding_vectors)

# Apply t-SNE for Dimensionality Reduction  
# Reduces high-dimensional face embeddings into a 2D space  
# for visualization while maintaining meaningful cluster relationships.  
tsne = TSNE(n_components=2, perplexity=min(10, len(embedding_vectors) - 1),
            learning_rate=50, random_state=42)
reduced_embeddings = tsne.fit_transform(embedding_vectors)

# Visualize Face Embedding Clusters  
# Each point represents a unique image, and colors indicate  
# different individuals based on folder names.  
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='tab10', s=80, alpha=0.8)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Face Embeddings Clustering using t-SNE")
plt.legend(title="Folders", loc="best", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("arcface_tsne.png", dpi=300)
plt.show()

# --- Confusion Matrix Generation ---
# Split data into training and testing sets to evaluate embedding quality
X_train, X_test, y_train, y_test = train_test_split(embedding_vectors, labels, test_size=0.3, random_state=42)

# Train a simple linear classifier on the embeddings
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Compute and Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel("Predicted Labels", fontsize=12, fontweight='bold')
plt.ylabel("True Labels", fontsize=12, fontweight='bold')
plt.title("Confusion Matrix (Buffalo_l Face Embeddings)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_buffalo.png", dpi=300)
plt.show()
