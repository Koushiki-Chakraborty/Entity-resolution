"""
=============================================================================
Step 3 — EPPO API Collector (Real Data)
Collect Type B (Synonyms) and Type C (Polysemy) pairs from EPPO

WHAT THIS DOES:
    Type B (Synonym pairs): Diseases with multiple names
    Type C (Polysemy pairs): Ambiguous names across different diseases
    
HOW TO RUN:
    python step3_eppo_collector_real.py
    
OUTPUT:
    eppo_pairs_collected.csv — Type B + Type C pairs

=============================================================================
"""

import pandas as pd
import requests
import time
import os
from pathlib import Path
import sys
import io

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_CSV = "../data/eppo_pairs_collected.csv"
ENV_FILE = "../../.env"

# Load EPPO API key from .env
def get_eppo_key():
    """Load EPPO_API_KEY from .env file"""
    possible_paths = [
        Path(ENV_FILE),
        Path(__file__).parent.parent.parent / ".env",
        Path.cwd().parent.parent / ".env",
    ]
    
    for env_path in possible_paths:
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('EPPO_API_KEY='):
                        key = line.split('=', 1)[1].strip().strip('"\'').strip()
                        if key and key != "":
                            return key
    return None

EPPO_API_KEY = get_eppo_key()
EPPO_BASE_URL = "https://data.eppo.int/api/v1"

POLYSEMY_TARGETS = [
    "rust", "blight", "wilt", "mosaic", "scab", "spot", "rot", "powdery mildew",
]

# ── EPPO FUNCTIONS ────────────────────────────────────────────────────────────

def collect_type_b_real():
    """Collect real Type B pairs from EPPO API"""
    print("\n[1/2] Collecting Type B synonym pairs from EPPO API...")
    pairs = []
    
    if not EPPO_API_KEY:
        print(f"  ⚠ No API key. Using mock data.")
        return get_type_b_mock()
    
    try:
        headers = {"X-API-Key": EPPO_API_KEY}
        
        # Query for diseases
        url = f"{EPPO_BASE_URL}/diseases"
        print(f"  Querying: {url}")
        
        response = requests.get(url, headers=headers, timeout=15, params={"limit": 100})
        print(f"  Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            diseases = data.get("data", data.get("results", []))[:50]
            print(f"  Found {len(diseases)} diseases")
            
            for disease in diseases:
                eppo_code = disease.get("eppo_code") or disease.get("code")
                main_name = disease.get("full_name") or disease.get("preferred_name") or disease.get("name", "")
                
                if not main_name or len(main_name) < 3:
                    continue
                
                alt_names = []
                
                # Try to get alternate names
                if eppo_code:
                    try:
                        url_names = f"{EPPO_BASE_URL}/diseases/{eppo_code}/names"
                        resp = requests.get(url_names, headers=headers, timeout=10)
                        if resp.status_code == 200:
                            names_data = resp.json().get("data", [])
                            alt_names = [n.get("name") for n in names_data if n.get("name") and n.get("name") != main_name][:2]
                        time.sleep(0.3)
                    except:
                        pass
                
                # Create pairs if we have alternates
                if alt_names:
                    context = disease.get("notes", disease.get("description", ""))[:200] or main_name
                    
                    for alt_name in alt_names:
                        pairs.append({
                            "name_a": main_name,
                            "name_b": alt_name,
                            "context_a": context,
                            "context_b": context,
                            "match": 1,
                            "source_a": "EPPO",
                            "source_b": "EPPO",
                            "pair_type": "B",
                            "context_quality_a": "good",
                            "context_quality_b": "good"
                        })
                    
                    if len(pairs) >= 50:
                        break
            
            print(f"  ✓ Collected {len(pairs)} Type B pairs from API")
            if pairs:
                return pairs
        
        print(f"  ✗ API returned status {response.status_code}")
    except Exception as e:
        print(f"  ✗ API error: {e}")
    
    print(f"  Falling back to mock data")
    return get_type_b_mock()


def collect_type_c_real():
    """Collect real Type C pairs from EPPO API"""
    print("\n[2/2] Collecting Type C polysemy pairs from EPPO API...")
    pairs = []
    
    if not EPPO_API_KEY:
        print(f"  ⚠ No API key. Using mock data.")
        return get_type_c_mock()
    
    try:
        headers = {"X-API-Key": EPPO_API_KEY}
        url = f"{EPPO_BASE_URL}/diseases"
        
        # Search for each polysemy target
        for target in POLYSEMY_TARGETS:
            try:
                print(f"  Searching for '{target}'...")
                params = {"q": target, "limit": 20}
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    diseases = data.get("data", data.get("results", []))[:4]
                    
                    if len(diseases) >= 2:
                        disease_pairs = []
                        for disease in diseases:
                            name = disease.get("full_name") or disease.get("preferred_name") or disease.get("name", "")
                            context = disease.get("notes", disease.get("description", ""))[:150] or target
                            if name and len(name) > 2:
                                disease_pairs.append((name, context))
                        
                        # Create pairs between different diseases
                        if len(disease_pairs) >= 2:
                            for i, (name_a, ctx_a) in enumerate(disease_pairs):
                                for name_b, ctx_b in disease_pairs[i+1:]:
                                    if name_a != name_b:
                                        pairs.append({
                                            "name_a": name_a,
                                            "name_b": name_b,
                                            "context_a": ctx_a,
                                            "context_b": ctx_b,
                                            "match": 0,
                                            "source_a": "EPPO",
                                            "source_b": "EPPO",
                                            "pair_type": "C",
                                            "context_quality_a": "good",
                                            "context_quality_b": "good"
                                        })
                            print(f"    → {len(disease_pairs)} diseases found")
                
                time.sleep(0.3)
                
                if len(pairs) >= 50:
                    break
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"  ✓ Collected {len(pairs)} Type C pairs from API")
        if pairs:
            return pairs
    except Exception as e:
        print(f"  ✗ API error: {e}")
    
    print(f"  Falling back to mock data")
    return get_type_c_mock()


def get_type_b_mock():
    """Mock Type B data"""
    sample_diseases = [
        {
            "name": "Panama disease",
            "alt": "Fusarium wilt of banana",
            "context": "Caused by Fusarium oxysporum f.sp. cubense"
        },
        {
            "name": "Late blight",
            "alt": "Phytophthora blight",
            "context": "Oomycete pathogen Phytophthora infestans"
        }
    ]
    
    pairs = []
    for d in sample_diseases:
        pairs.append({
            "name_a": d["name"],
            "name_b": d["alt"],
            "context_a": d["context"],
            "context_b": d["context"],
            "match": 1,
            "source_a": "EPPO_Mock",
            "source_b": "EPPO_Mock",
            "pair_type": "B",
            "context_quality_a": "good",
            "context_quality_b": "good"
        })
    
    return pairs


def get_type_c_mock():
    """Mock Type C data"""
    polysemy_groups = [
        {
            "target": "rust",
            "diseases": [
                ("Wheat rust", "Puccinia striiformis causes stripe rust"),
                ("Corn rust", "Puccinia sorghi causes common rust")
            ]
        }
    ]
    
    pairs = []
    for group in polysemy_groups:
        diseases = group["diseases"]
        for i, (name_a, ctx_a) in enumerate(diseases):
            for name_b, ctx_b in diseases[i+1:]:
                pairs.append({
                    "name_a": name_a,
                    "name_b": name_b,
                    "context_a": ctx_a,
                    "context_b": ctx_b,
                    "match": 0,
                    "source_a": "EPPO_Mock",
                    "source_b": "EPPO_Mock",
                    "pair_type": "C",
                    "context_quality_a": "good",
                    "context_quality_b": "good"
                })
    
    return pairs


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(" EPPO API Collector — Type B & C Pairs")
    print("=" * 70)
    
    if EPPO_API_KEY:
        print(f"\n✅ EPPO API key loaded ({EPPO_API_KEY[:20]}...)")
    else:
        print(f"\n⚠ No EPPO API key found. Using mock data.")
    
    # Collect both types
    pairs_b = collect_type_b_real()
    pairs_c = collect_type_c_real()
    
    all_pairs = pairs_b + pairs_c
    df = pd.DataFrame(all_pairs)
    
    # Save
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    print(f"\n✅ Saved {len(df)} pairs to {OUTPUT_CSV}")
    print(f"  Type B (synonyms): {(df['pair_type']=='B').sum()}")
    print(f"  Type C (polysemy): {(df['pair_type']=='C').sum()}")


if __name__ == "__main__":
    main()
