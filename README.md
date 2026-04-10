# 🎓 Face Recognition Based Attendance System

An automated attendance system that uses **deep learning-based face recognition** to detect and identify students from classroom photographs. Built with **InsightFace (ArcFace)** for state-of-the-art face detection and recognition.

---

## ✨ Features

- **Multi-face detection** — Detects and recognizes multiple students in a single classroom photo
- **ArcFace (GlintR100)** — Uses the most accurate face recognition model available (99.83% on LFW benchmark)
- **SCRFD detector** — Fast, robust face detection optimized for crowded scenes
- **Data augmentation** — Augments training images for better generalization
- **t-SNE visualization** — Visualizes face embedding clusters to verify training quality
- **Annotated output** — Generates images with bounding boxes and name labels for each recognized student

---

## 🏗️ Architecture

| Component | Model | Details |
|-----------|-------|---------|
| **Face Detection** | SCRFD 10G | Fast multi-face detector, handles small/distant faces |
| **Face Recognition** | ArcFace GlintR100 | ResNet-100 trained on Glint360K (360K identities) |
| **Model Pack** | `antelopev2` | InsightFace's highest accuracy model bundle |
| **Embeddings** | 512-dimensional | L2-normalized cosine similarity matching |

---

## 📁 Project Structure

```
mini-project/
├── dataset/                    # Raw face images (organized by person name)
│   ├── abhiraj/
│   ├── abhisekh/
│   ├── arijit/
│   └── ... (22 students)
├── test-images/                # Classroom photos to run recognition on
├── output/                     # Annotated output images with bounding boxes
│
├── aug.py                      # Data augmentation script
├── train_arcface.py            # 🏋️ Training script (ArcFace + antelopev2)
├── recognize_arcface.py        # 🔍 Recognition script (ArcFace + antelopev2)
│
├── train_antelope.py           # Alternative training (antelopev2, older version)
├── train_buffalo.py            # Alternative training (buffalo_l model)
├── recognise_antelope.py       # Alternative recognition (antelopev2)
├── recognize_buffalo.py        # Alternative recognition (buffalo_l)
├── recognize1.py               # Legacy recognition script
│
├── arcface_tsne.png            # t-SNE cluster visualization
├── similarity_distribution.png # Similarity score distribution
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- macOS (Apple Silicon M-series) / Linux / Windows

### 1. Clone the Repository

```bash
git clone https://github.com/Monaswi0104/Mini_Project.git
cd Mini_Project
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

The InsightFace `antelopev2` model will auto-download on first run, or you can manually place it:

```
~/.insightface/models/antelopev2/
├── scrfd_10g_bnkps.onnx    # Face detector
├── glintr100.onnx          # Face recognizer (ArcFace)
├── 1k3d68.onnx             # 3D landmarks
├── 2d106det.onnx           # 2D landmarks
└── genderage.onnx          # Gender/age estimation
```

---

## 📋 Usage

### Step 1: Prepare Dataset

Organize face images in the `dataset/` folder:

```
dataset/
├── student_name_1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── student_name_2/
│   ├── 1.jpg
│   └── ...
```

Each student should have **8-10 clear face images** from different angles.

### Step 2: Augment Data (Optional)

```bash
python aug.py
```

This generates augmented images in the `augmented-dataset/` folder.

### Step 3: Train the Model

```bash
python train_arcface.py
```

This processes all face images, extracts 512-dimensional embeddings using ArcFace, and saves them to `face_embeddings_arcface.pkl`.

### Step 4: Run Recognition

Place classroom photos in the `test-images/` folder, then run:

```bash
python recognize_arcface.py
```

Annotated output images will be saved in the `output/` folder with:
- 🟩 **Green box** — Recognized student (with name and similarity score)
- 🟥 **Red box** — Unknown person

---

## ⚙️ Configuration

### Recognition Threshold

In `recognize_arcface.py`, adjust the similarity threshold (line 81):

```python
if similarity > best_similarity and similarity > 0.45:  # Adjust this value
```

| Threshold | Effect |
|-----------|--------|
| `0.65` | Strict — fewer false positives, may miss some students |
| `0.55` | Balanced — good for most use cases |
| `0.45` | Lenient — catches more students, may have false matches |

### Detection Resolution

In `recognize_arcface.py`, adjust `det_size` for different photo resolutions:

```python
app.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.3)
```

- `(640, 640)` — Faster, good for close-up photos
- `(1280, 1280)` — Catches small/distant faces in classroom photos
- `(1920, 1920)` — Maximum detection for very large group photos

---

## 📊 Results

### t-SNE Embedding Clusters

The training script generates a t-SNE visualization showing how well the model separates different students' face embeddings in 2D space.

### Sample Output

Recognition generates annotated images with bounding boxes, names, and confidence scores for each detected face.

---

## 🛠️ Tech Stack

- **[InsightFace](https://github.com/deepinsight/insightface)** — Face detection & recognition framework
- **[ONNX Runtime](https://onnxruntime.ai/)** — Optimized model inference
- **[OpenCV](https://opencv.org/)** — Image processing
- **[scikit-learn](https://scikit-learn.org/)** — t-SNE visualization
- **[Pillow](https://pillow.readthedocs.io/)** — Image annotation & drawing
- **[PyTorch](https://pytorch.org/)** — Deep learning framework

---

## 👥 Authors

- **Monaswi Kumar Bharadwaj** — [Monaswi0104](https://github.com/Monaswi0104)

---

## 📄 License

This project is for academic/educational purposes.
