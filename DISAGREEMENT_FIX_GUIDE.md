# MANUAL FIX GUIDE — 40 Label Disagreements

## How to Use This Guide

1. **Read through the recommendations below**
2. **Open `data/dataset_fixed.csv` in VS Code** (as CSV or text)
3. **For each row marked "FIX"**, find it by the `name_a` and `name_b` pair and change the `match` column
4. **Save the file as `data/dataset_fixed_reviewed.csv`** (or overwrite `dataset_fixed.csv`)
5. **Run step2** with the corrected file

---

## 40 Disagreements — Categorized with Recommendations

### STRONG FIX RECOMMENDATIONS (High Confidence)

These are labeling errors you should fix:

| name_a                             | name_b                    | pair_type | Your Label | LLM Label | **Recommendation** | Reason                                  |
| ---------------------------------- | ------------------------- | --------- | ---------- | --------- | ------------------ | --------------------------------------- |
| maize streak virus                 | maize streak virus (msv)  | C         | 0          | 1         | **FIX to 1** ✓     | Same virus; MSV is just abbreviation    |
| maize streak virus (msv)           | maize streak disease      | C         | 0          | 1         | **FIX to 1** ✓     | MSV causes this disease; same entity    |
| sclerotinia sclerotiorum           | sclerotinia               | C         | 0          | 1         | **FIX to 1** ✓     | Genus vs species of same fungus         |
| tomato septoria leaf spot          | leaf spot of tomato       | C         | 0          | 1         | **FIX to 1** ✓     | Just different word order; same disease |
| septoria leaf spot                 | tomato septoria leaf spot | C         | 0          | 1         | **FIX to 1** ✓     | Generic vs specific; same pathogen      |
| feline syncytial virus             | bovine syncytial virus    | A         | 1          | 0         | **FIX to 0** ✓     | Different host viruses; not the same    |
| cowpea chlorotic mottle bromovirus | brome mosaic bromovirus   | B         | 1          | 0         | **FIX to 0** ✓     | Different bromovirus species            |

---

### LIKELY CORRECT AS-IS (Your label is probably right)

Keep these unchanged — LLM appears to be wrong:

| name_a                            | name_b                             | pair_type | Your Label | LLM Label | **Recommendation** | Reason                                                   |
| --------------------------------- | ---------------------------------- | --------- | ---------- | --------- | ------------------ | -------------------------------------------------------- |
| erwinia carotovora                | erwinia aroideae                   | A         | 1          | 0         | **KEEP 1**         | Same bacterial genus                                     |
| hantaan virus                     | hantavirus                         | B         | 1          | 0         | **KEEP 1**         | Hantavirus genus includes Hantaan                        |
| puumala virus                     | hantavirus                         | B         | 1          | 0         | **KEEP 1**         | Both are Hantavirus genus                                |
| prrs virus                        | lelystad virus                     | A         | 1          | 0         | **KEEP 1**         | PRRS = Porcine Reprod. Resp. Synd.; Lelystad is old name |
| fowl pest virus                   | newcastle disease virus            | A         | 1          | 0         | **KEEP 1**         | Newcastle disease IS fowl pest                           |
| mucosal disease virus             | pestivirus                         | B         | 1          | 0         | **KEEP 1**         | MD virus IS a type of pestivirus                         |
| syncytial viruses                 | feline syncytial virus             | A         | 1          | 0         | **KEEP 1**         | Generic vs specific; related enough                      |
| bovine syncytial virus            | spumavirus                         | B         | 1          | 0         | **KEEP 1**         | BSV is a type of spumavirus                              |
| brome mosaic bromovirus           | bromovirus                         | A         | 1          | 0         | **KEEP 1**         | Specific vs genus; semantically same                     |
| protozoans                        | protozoa                           | A         | 1          | 0         | **KEEP 1**         | Plural vs singular of same concept                       |
| pacific islands (trust territory) | trust territory of pacific islands | A         | 1          | 0         | **KEEP 1**         | Same location, just phrased differently                  |

---

### PROBABLY CORRECT AS-IS (Keep your labels)

These align with domain logic:

| name_a              | name_b       | pair_type | Your Label | LLM Label | **Recommendation** | Reason                                 |
| ------------------- | ------------ | --------- | ---------- | --------- | ------------------ | -------------------------------------- |
| corn smut           | pleioblastus | D         | 0          | 1         | **KEEP 0**         | Corn fungus vs bamboo genus; different |
| rabbit-like viruses | lyssavirus   | B         | 1          | 0         | **KEEP 1**         | Rabies-like ARE Lyssavirus             |
| potyvirus           | wsmv         | D         | 0          | 1         | **FIX to 1**       | WSMV IS a potyvirus genus              |

---

### EDGE CASES (Requires your judgment / domain knowledge)

These are genuinely ambiguous — make best judgment:

| name_a                     | name_b                              | pair_type | Your Label | **Notes**                                          |
| -------------------------- | ----------------------------------- | --------- | ---------- | -------------------------------------------------- |
| kashmir bee virus          | iflavirus                           | D         | 0          | Could be instance of; need pathogen database       |
| alfalfa mosaic virus group | rna tumour viruses                  | D         | 0          | Group vs general type; likely different            |
| fusarium wilt              | orbivirus                           | D         | 0          | Fungal disease vs virus genus; clearly different   |
| fusarium sporotrichiella   | fusarium sporotrichioides           | A         | 1          | Different Fusarium species; taxonomically related  |
| grape powdery mildew       | powdery mildew                      | C         | 0          | Different hosts likely = different diseases        |
| isariopsis leaf spot       | grape leaf spot                     | C         | 0          | Different fungi, different crops; likely different |
| closterovirus              | beet necrotic yellow vein furovirus | D         | 0          | Different virus families; keep as-is               |
| rhizomania virus           | plant rhabdovirus group b           | D         | 0          | Different virus types; keep as-is                  |
| yellow shoot disease       | black sigatoka                      | D         | 0          | Different diseases entirely; keep as-is            |
| fusarium wilt of cucumber  | panama disease                      | D         | 0          | Different crops/pathogens; keep as-is              |
| erwinia carotovora         | tomato yellow leaf curl virus       | D         | 0          | Bacteria vs virus; keep as-is                      |
| maize streak disease       | diabrotica undecimpunctata          | D         | 0          | Virus vs beetle; keep as-is                        |
| yellow shoot disease       | proterorhinus marmoratus            | D         | 0          | Different kingdoms; keep as-is                     |
| rna tumour viruses         | potyvirus group                     | D         | 0          | Different virus groups; keep as-is                 |
| septoria leaf spot         | leaf spot of tomato                 | C         | 0          | Different crops; might be different pathogens      |

---

## Summary of Recommended Changes

**TOTAL FIXES: 8 rows**

```
1. maize streak virus ↔ maize streak virus (msv)           → match=1
2. maize streak virus (msv) ↔ maize streak disease        → match=1
3. sclerotinia sclerotiorum ↔ sclerotinia                 → match=1
4. tomato septoria leaf spot ↔ leaf spot of tomato        → match=1
5. septoria leaf spot ↔ tomato septoria leaf spot         → match=1
6. feline syncytial virus ↔ bovine syncytial virus        → match=0
7. cowpea chlorotic mottle bromovirus ↔ brome mosaic bromovirus → match=0
8. potyvirus ↔ wsmv                                        → match=1
```

All others: **KEEP AS-IS** (your labels are correct)

---

## How to Make the Fixes in VS Code

### Option 1: Using Find & Replace

1. Open `data/dataset_fixed.csv` in VS Code
2. For each pair above, use Ctrl+F to find the exact row
3. Edit the `match` column value directly
4. Save as `dataset_fixed_reviewed.csv`

### Option 2: Use Python script

```bash
cd dataset_v2_builder
python fix_disagreements.py
```

(I can create this script for you if you want!)

### Option 3: Manual editing

1. Open spreadsheet app (Excel, Google Sheets, LibreOffice Calc)
2. Load `dataset_fixed.csv`
3. Find rows by name pairs
4. Change `match` column
5. Export as CSV

---

## Next Steps

1. **Make the 8 fixes** above
2. **Save as** `dataset_fixed_reviewed.csv`
3. **Run step2** (Wikipedia scraper):
   ```bash
   python step2_scrape_types.py --input data/dataset_fixed_reviewed.csv --output data/scraped_types.csv
   ```
4. Continue with steps 3 and 4

Let me know which approach you prefer and I'll help you execute it!
