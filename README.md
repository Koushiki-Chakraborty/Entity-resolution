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
