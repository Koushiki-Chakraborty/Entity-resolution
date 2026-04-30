# Entity-resolution

## Tagging Ablation Study
## Agricultural Disease Entity Resolution

---

## 🎯 Objective
To evaluate the impact of structured tagging on embedding quality.

We test whether adding domain-specific tags like `[DISEASE]` and `[CONTEXT]` improves similarity separation between:
- Matching disease pairs
- Non-matching pairs (including disease vs pest)

---

## ⚙️ Experiment Setup

We compare three input formats using a base model (`all-MiniLM-L6-v2`):

1. **Name Only**
   - Uses only the disease name  
   - No context, no tags  

2. **Name + Context**
   - Concatenates disease name with description  
   - No tags  

3. **Ditto-Style Tagged Input*

# 🌱 Agricultural Entity Resolution Pipeline

This project builds a **domain-adapted entity resolution system** for agriculture (diseases, pests, viruses, etc.) using Sentence Transformers + FAISS.

It follows a **3-step pipeline**:

* **Step 4 → Fine-tune encoder**
* **Step 5 → Find optimal similarity threshold**
* **Step 6 → Deploy FAISS-based resolution system**

---

# 🚀 Pipeline Overview

## 🔹 Step 4: Fine-tuning the Encoder

Train a domain-adapted embedding model using:

* Entity names + context
* Entity type tags (DISEASE, PEST, etc.)
* Soft labels (`lambda_val`) + hard labels (`match`)

### ▶ Run

```bash
python step4_finetune.py
```

### 📥 Input

* `Dataset/dataset_production_ready.csv`

### 📤 Output

* `./plant-disease-encoder/` → trained model
* `test_set.csv` → held-out test data

---

## 🏷️ Tagging Strategy (Core Innovation)

This project uses **Ditto-style structured tagging** to improve embedding quality.

### 🔹 Input Format Used

```text
[DISEASE] Leaf Blight [/DISEASE] 
[CONTEXT] fungal infection causing leaf damage in rice [/CONTEXT]
```

### 🔹 Why Tagging Matters

Tagging helps the model:

* Understand **entity boundaries**
* Differentiate between **entity types (disease vs pest)**
* Give more importance to **context vs name**
* Reduce confusion in noisy real-world data

---

## 🧪 Tagging Ablation Study

We experimentally evaluate how tagging affects performance.

### 📊 Compared Input Variants

| Method             | Description                  |
| ------------------ | ---------------------------- |
| Name Only          | Only entity name             |
| Name + Context     | Plain concatenation          |
| Ditto-Style Tagged | Structured tags (our method) |

---

### 🧠 Observations

* **Name Only** → Poor performance (no context understanding)
* **Name + Context** → Better, but still ambiguous
* **Tagged Input** → Best performance

---

### 🏆 Conclusion

Ditto-style tagging:

* Improves **semantic representation**
* Creates **clear similarity separation**
* Increases **F1 score**
* Helps model distinguish:

  * Disease vs Pest
  * Similar names with different meanings

---

## 🔹 Step 5: Threshold Sweep + Evaluation

Find the **best cosine similarity threshold** for classification.

### ▶ Run

```bash
python step5_evaluate.py
```

### 📥 Input

* `./plant-disease-encoder/`
* `test_set.csv`

### 📤 Output

* `threshold_results.csv`
* Printed:

  * Best threshold
  * F1 score
  * Precision / Recall
  * Confusion matrix

---

### ⚙️ What it does

* Computes cosine similarity between entity pairs
* Sweeps thresholds from **0.1 → 0.9**
* Selects threshold with **maximum F1 score**

---

## 🔹 Step 6: FAISS Blocking + Entity Resolution

Deploy a **fast search system** using FAISS.

### ▶ Install

```bash
pip install faiss-cpu
```

### ▶ Run

```bash
python step6_blocking.py
```

### 📥 Input

* Trained model
* Dataset
* `threshold_results.csv`

### 📤 Output

* `disease_index.faiss` → vector index
* `disease_index_metadata.csv` → entity metadata

---

# ⚡ Features of Step 6

## 🔍 1. Knowledge Base Creation

* Extracts unique canonical entities
* Keeps best context per entity

## ⚡ 2. FAISS Index

* Converts entities into embeddings
* Enables **fast similarity search (Top-K retrieval)**

## 🎯 3. Entity Resolution

For a given input:

* Finds top candidates
* Applies threshold
* Returns:

  * ✅ MATCHED entity
  * ❌ NEW entity

---

## 🧪 4. Pipeline Evaluation

* Tests full system on test set
* Reports:

  * Recall
  * Per-entity-type performance

---

## 💬 5. Interactive Mode

You can test the system manually:

```text
Entity name     : leaf blight
Context         : fungal infection in rice
Entity type     : DISEASE
```

---

# 📊 Final Outputs

| File                         | Description              |
| ---------------------------- | ------------------------ |
| `plant-disease-encoder/`     | Trained model            |
| `test_set.csv`               | Evaluation dataset       |
| `threshold_results.csv`      | Threshold tuning results |
| `disease_index.faiss`        | FAISS index              |
| `disease_index_metadata.csv` | Entity metadata          |

---

# 🔮 Next Step

➡ Run:

```bash
python step6_blocking.py
```

➡ Use the system for:

* Real-time entity resolution
* Knowledge base expansion
* Agricultural AI applications

---

# 👩‍💻 Author

Built as part of an **Agricultural Entity Resolution System**
Focused on real-world noisy data + production-ready pipeline.

---
