import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont

# Define Paths for Input and Output
TEST_FOLDER = "test-images"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load Stored Face Embeddings
embeddings_path = "face_embeddings_arcface.pkl"
try:
    with open(embeddings_path, "rb") as f:
        known_faces = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {embeddings_path}. Please run train_arcface.py first.")
    exit(1)

for name in known_faces:
    known_faces[name] /= np.linalg.norm(known_faces[name])  # Normalize vectors

# Initialize InsightFace with ArcFace (antelopev2)
# Uses SCRFD for fast multi-face detection + GlintR100 for recognition
print("Loading InsightFace ArcFace model (antelopev2)...")
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
# det_size=(1280,1280) → higher resolution catches small/distant faces in classroom photos
# det_thresh=0.3 → lower threshold to detect partially occluded or side-profile faces
app.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.3)
print("Model loaded successfully!")

def cosine_similarity(a, b):
    """Cosine similarity calculates similarity score between -1 and 1."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Process Each Image in the Test Folder
for image_name in os.listdir(TEST_FOLDER):
    if image_name.startswith('.'):
        continue

    image_path = os.path.join(TEST_FOLDER, image_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Skipping {image_name}, cannot read file.")
        continue

    # InsightFace works with BGR (OpenCV default), so no conversion needed for detection
    faces = app.get(img)

    # Convert to RGB PIL Image for drawing annotations
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Load Font for Name Labels
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()

    print(f"\nProcessing {image_name}:")

    print(f" - Total faces detected: {len(faces)}")

    if len(faces) == 0:
        print(" - No faces detected.")
    else:
        for face in faces:
            embedding = face.normed_embedding
            embedding /= np.linalg.norm(embedding)  # Normalize embedding vector
            bbox = face.bbox.astype(int)

            # Compare with Stored Embeddings to Identify the Person
            best_match = "Unknown"
            best_similarity = 0.00

            for name, known_emb in known_faces.items():
                similarity = cosine_similarity(embedding, known_emb)

                # ArcFace embeddings are very discriminative;
                # 0.55 is a good threshold (adjust between 0.45 - 0.65 depending on strictness)
                if similarity > best_similarity and similarity > 0.45:
                    best_match = name.capitalize()
                    best_similarity = similarity

            # Draw Bounding Box and Name Label
            box_color = "red" if best_match == "Unknown" else "green"
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=box_color, width=5)

            text_position = (bbox[0], bbox[1] - 60)
            text_color = "red" if best_match == "Unknown" else "white"

            similarity_display = f"{best_similarity:.2f}" if best_match != "Unknown" else "0.00"
            draw.text(text_position, f"{best_match} ({similarity_display})", fill=text_color, font=font)

            print(f" - Detected: {best_match} (Similarity: {similarity_display})")

    # Save the Annotated Image
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{image_name}")
    pil_img.save(output_path)
    print(f"Processed {image_name}, saved to {output_path}")

print("\nRecognition complete! Check the 'output' folder.")
