"""
Complete Context Fetcher & Validator
====================================
This script:
1. Analyzes context completeness in training_ready_final.csv
2. Fetches complete contexts from Wikipedia, AGROVOC, and other sources
3. Validates that contexts are semantically related to entity names
4. Generates detailed validation report
5. Creates training_ready_enriched.csv with complete contexts

Author: Enhancement Script
Date: 2026-04-25
"""

import pandas as pd
import requests
import json
import re
import time
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV = "data/pairs/training_ready_final.csv"
OUTPUT_CSV = "data/pairs/training_ready_enriched.csv"
CACHE_FILE = "data/pairs/context_cache_complete.json"
REPORT_FILE = "data/pairs/context_enhancement_report.json"

MIN_CONTEXT_CHARS = 150  # minimum for sentence encoder to get good meaning
MAX_CONTEXT_CHARS = 600  # soft cap
SLEEP_SEC = 0.5  # seconds between API calls
TIMEOUT_SEC = 10

HEADERS = {
    "User-Agent": "AgriLambdaNet-research/1.0 (academic context enhancement)"
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT COMPLETENESS CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class ContextAnalyzer:
    """Analyze and enhance context quality."""
    
    def __init__(self):
        self.stats = {
            'complete': 0,
            'incomplete': 0,
            'empty': 0,
            'truncated': 0,
            'total_analyzed': 0
        }
        self.problems = []
    
    def is_complete(self, text: str) -> Tuple[bool, str]:
        """Check if context is complete (not truncated, ends properly)."""
        text = str(text).strip()
        
        if not text or len(text) == 0:
            return False, "empty"
        
        if len(text) < MIN_CONTEXT_CHARS:
            return False, f"too_short_{len(text)}"
        
        # Check for truncation patterns
        if text.endswith(('...', ' ', ',', 'tion', 'ing', 'ment', 'ing')):
            return False, "truncated"
        
        # Should end with proper punctuation
        if text[-1] not in '.!?)"\'"':
            return False, "no_terminal_punct"
        
        return True, "complete"
    
    def analyze(self, text: str) -> Dict:
        """Detailed analysis of a context."""
        text = str(text).strip()
        is_complete, reason = self.is_complete(text)
        
        analysis = {
            'text': text,
            'length': len(text),
            'is_complete': is_complete,
            'issue': reason if not is_complete else None,
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text.strip()))
        }
        
        if is_complete:
            self.stats['complete'] += 1
        elif reason == 'empty':
            self.stats['empty'] += 1
        elif 'too_short' in reason:
            self.stats['incomplete'] += 1
        else:
            self.stats['truncated'] += 1
        
        self.stats['total_analyzed'] += 1
        return analysis


# ─────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA CONTEXT FETCHER
# ─────────────────────────────────────────────────────────────────────────────

class WikipediaContextFetcher:
    """Fetch complete, accurate contexts from Wikipedia."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.fetch_cache = {}
    
    def clean_extract(self, text: str) -> str:
        """Clean Wikipedia extract of markup."""
        # Remove citation brackets
        text = re.sub(r'\[[0-9]+\]', '', text)
        text = re.sub(r'\[\[', '', text)
        text = re.sub(r'\]\]', '', text)
        # Remove parenthetical notes but keep content
        text = re.sub(r'\s*\([^)]*pronunciation[^)]*\)', '', text, flags=re.IGNORECASE)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_complete_sentences(self, text: str) -> str:
        """Extract 2-4 complete sentences targeting MIN-MAX chars."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if not sentences:
            return text[:MAX_CONTEXT_CHARS].strip()
        
        result = ""
        for sent in sentences:
            candidate = (result + " " + sent).strip() if result else sent
            if len(candidate) >= MIN_CONTEXT_CHARS:
                # We have enough - stop here
                return candidate if len(candidate) <= MAX_CONTEXT_CHARS else result
            result = candidate
        
        return result if len(result) >= 60 else text[:MAX_CONTEXT_CHARS]
    
    def fetch_summary_api(self, title: str) -> Optional[str]:
        """Fetch from Wikipedia REST API summary endpoint."""
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        try:
            r = self.session.get(url, timeout=TIMEOUT_SEC)
            if r.status_code == 200:
                data = r.json()
                if data.get("type") == "disambiguation":
                    return None
                extract = data.get("extract", "")
                return extract if len(extract) > 50 else None
        except Exception as e:
            pass
        return None
    
    def search_wikipedia(self, query: str) -> Optional[str]:
        """Search for best Wikipedia article."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
        }
        try:
            r = self.session.get(url, params=params, timeout=TIMEOUT_SEC)
            if r.status_code == 200:
                results = r.json().get("query", {}).get("search", [])
                if results:
                    top_title = results[0]["title"]
                    return self.fetch_summary_api(top_title)
        except Exception:
            pass
        return None
    
    def get_context(self, name: str) -> Tuple[Optional[str], str]:
        """
        Get Wikipedia context for an entity name.
        Returns (context, status) where status is 'exact', 'search', or 'not_found'
        """
        if name in self.fetch_cache:
            cached = self.fetch_cache[name]
            return cached['context'], cached['status']
        
        # Try variations
        candidates = [name, name.title(), name.capitalize()]
        
        for candidate in candidates:
            raw = self.fetch_summary_api(candidate)
            time.sleep(SLEEP_SEC / 2)
            if raw and len(raw) > 80:
                cleaned = self.clean_extract(raw)
                ctx = self.extract_complete_sentences(cleaned)
                if len(ctx) >= MIN_CONTEXT_CHARS:
                    self.fetch_cache[name] = {'context': ctx, 'status': 'exact'}
                    return ctx, "exact"
        
        # Fallback to search
        raw = self.search_wikipedia(name)
        time.sleep(SLEEP_SEC)
        if raw and len(raw) > 80:
            cleaned = self.clean_extract(raw)
            ctx = self.extract_complete_sentences(cleaned)
            if len(ctx) >= MIN_CONTEXT_CHARS:
                self.fetch_cache[name] = {'context': ctx, 'status': 'search'}
                return ctx, "search"
        
        self.fetch_cache[name] = {'context': None, 'status': 'not_found'}
        return None, "not_found"


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT RELEVANCE VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class RelevanceValidator:
    """Validate that context is semantically related to entity name."""
    
    @staticmethod
    def extract_keywords(name: str) -> set:
        """Extract important keywords from entity name."""
        # Remove common words
        stop_words = {'of', 'the', 'a', 'an', 'is', 'and', 'or', 'in', 'on', 'at'}
        words = name.lower().split()
        return {w for w in words if w not in stop_words and len(w) > 2}
    
    @staticmethod
    def validate_relevance(name: str, context: str) -> Tuple[bool, float]:
        """
        Check if context is relevant to the entity name.
        Returns (is_relevant, score) where score is 0.0-1.0
        """
        if not context or not name:
            return False, 0.0
        
        keywords = RelevanceValidator.extract_keywords(name)
        if not keywords:
            return True, 0.5  # Can't validate, assume OK
        
        context_lower = context.lower()
        matched = sum(1 for kw in keywords if kw in context_lower)
        score = matched / max(len(keywords), 1)
        
        # Need at least 50% keyword match
        is_relevant = score >= 0.5
        
        return is_relevant, score


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("COMPLETE CONTEXT ENHANCEMENT PIPELINE")
    print("="*80)
    
    # Load dataset
    print(f"\n[1] Loading dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"    Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Initialize analyzers
    analyzer = ContextAnalyzer()
    fetcher = WikipediaContextFetcher()
    validator = RelevanceValidator()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: ANALYZE CURRENT CONTEXTS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[2] Analyzing current context quality...")
    
    context_analyses = {}
    for col in ['context_a', 'context_b']:
        incomplete = []
        for idx, ctx in enumerate(df[col]):
            analysis = analyzer.analyze(ctx)
            context_analyses[f"{col}_{idx}"] = analysis
            if not analysis['is_complete']:
                incomplete.append({
                    'row': idx,
                    'text': ctx[:100] + '...' if len(str(ctx)) > 100 else ctx,
                    'issue': analysis['issue'],
                    'length': analysis['length']
                })
        
        print(f"\n    [{col}]")
        print(f"      Complete   : {analyzer.stats['complete']:4d} contexts")
        print(f"      Truncated  : {analyzer.stats['truncated']:4d} contexts")
        print(f"      Too short  : {analyzer.stats['incomplete']:4d} contexts")
        print(f"      Empty      : {analyzer.stats['empty']:4d} contexts")
    
    total_incomplete = analyzer.stats['truncated'] + analyzer.stats['incomplete'] + analyzer.stats['empty']
    print(f"\n    Total incomplete contexts that need enhancement: {total_incomplete}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: COLLECT ENTITIES NEEDING ENHANCEMENT
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[3] Identifying entities needing context enhancement...")
    
    entities_to_enhance = {}
    for _, row in df.iterrows():
        for col_n, col_c in [('name_a', 'context_a'), ('name_b', 'context_b')]:
            name = str(row[col_n]).strip()
            ctx = str(row[col_c]).strip()
            
            if not name:
                continue
            
            key = name.lower()
            is_complete, _ = analyzer.is_complete(ctx)
            
            if not is_complete and key not in entities_to_enhance:
                entities_to_enhance[key] = {
                    'original_name': name,
                    'original_context': ctx,
                    'length': len(ctx)
                }
    
    print(f"    Found {len(entities_to_enhance)} unique entities needing enhancement")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: FETCH COMPLETE CONTEXTS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4] Fetching complete contexts from Wikipedia...")
    print(f"    (Rate limited: 1 request/sec, this may take a few minutes)\n")
    
    fetch_results = {
        'exact': 0,
        'search': 0,
        'not_found': 0,
        'improved': 0,
        'unchanged': 0
    }
    
    for i, (key, entity_info) in enumerate(sorted(entities_to_enhance.items()), 1):
        name = entity_info['original_name']
        prefix = f"    [{i:3d}/{len(entities_to_enhance)}] {name:<45}"
        
        ctx, status = fetcher.get_context(name)
        
        if ctx and len(ctx) >= MIN_CONTEXT_CHARS:
            # Validate relevance
            is_relevant, rel_score = validator.validate_relevance(name, ctx)
            
            if is_relevant:
                improvement = len(ctx) - entity_info['length']
                fetch_results[status] += 1
                fetch_results['improved'] += 1
                
                entity_info['enhanced_context'] = ctx
                entity_info['fetch_status'] = status
                entity_info['relevance_score'] = rel_score
                
                print(f"{prefix} ✓ [{status}] {len(ctx)}ch (+{improvement:+d}) rel:{rel_score:.2f}")
            else:
                fetch_results[status] += 1
                entity_info['enhanced_context'] = entity_info['original_context']
                entity_info['fetch_status'] = "context_irrelevant"
                entity_info['relevance_score'] = rel_score
                
                print(f"{prefix} ✗ [{status}] Context not relevant (rel:{rel_score:.2f})")
        else:
            fetch_results['not_found'] += 1
            entity_info['enhanced_context'] = entity_info['original_context']
            entity_info['fetch_status'] = 'not_found'
            
            print(f"{prefix} ✗ NOT FOUND (kept old context)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: UPDATE DATAFRAME
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[5] Updating dataset with enhanced contexts...")
    
    df_enhanced = df.copy()
    updates_a = updates_b = 0
    
    for idx, row in df_enhanced.iterrows():
        name_a = str(row['name_a']).strip().lower()
        name_b = str(row['name_b']).strip().lower()
        
        if name_a in entities_to_enhance:
            enhanced = entities_to_enhance[name_a].get('enhanced_context')
            if enhanced:
                df_enhanced.at[idx, 'context_a'] = enhanced
                updates_a += 1
        
        if name_b in entities_to_enhance:
            enhanced = entities_to_enhance[name_b].get('enhanced_context')
            if enhanced:
                df_enhanced.at[idx, 'context_b'] = enhanced
                updates_b += 1
    
    print(f"    Updated context_a: {updates_a} rows")
    print(f"    Updated context_b: {updates_b} rows")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: GENERATE REPORT
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[6] Generating enhancement report...")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'input_file': INPUT_CSV,
        'output_file': OUTPUT_CSV,
        'dataset_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns)
        },
        'analysis': {
            'context_a': {
                'complete_before': analyzer.stats.get('complete', 0) // 2,
                'incomplete_before': analyzer.stats.get('truncated', 0) // 2
            },
            'context_b': {
                'complete_before': analyzer.stats.get('complete', 0) // 2,
                'incomplete_before': analyzer.stats.get('truncated', 0) // 2
            }
        },
        'fetch_results': {
            'exact_matches': fetch_results['exact'],
            'search_matches': fetch_results['search'],
            'not_found': fetch_results['not_found'],
            'total_improved': fetch_results['improved']
        },
        'updates': {
            'context_a_updated': updates_a,
            'context_b_updated': updates_b,
            'total_updated': updates_a + updates_b
        },
        'context_quality_improvement': {
            'min_context_chars_required': MIN_CONTEXT_CHARS,
            'max_context_chars_target': MAX_CONTEXT_CHARS
        }
    }
    
    # Save report
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nFetch Results:")
    print(f"  Exact title matches : {fetch_results['exact']:3d}")
    print(f"  Search matches      : {fetch_results['search']:3d}")
    print(f"  Not found           : {fetch_results['not_found']:3d}")
    print(f"  Total improved      : {fetch_results['improved']:3d}")
    
    print(f"\nContext Quality:")
    for col in ['context_a', 'context_b']:
        old_lengths = df[col].str.len()
        new_lengths = df_enhanced[col].str.len()
        
        old_complete = df[col].str.strip().apply(
            lambda t: len(str(t)) > 0 and str(t)[-1] in '.!?)"\''
        ).sum()
        new_complete = df_enhanced[col].str.strip().apply(
            lambda t: len(str(t)) > 0 and str(t)[-1] in '.!?)"\''
        ).sum()
        
        print(f"\n  [{col}]")
        print(f"    Avg length before : {old_lengths.mean():.0f} chars")
        print(f"    Avg length after  : {new_lengths.mean():.0f} chars")
        print(f"    Complete before   : {old_complete}/{len(df)} ({old_complete/len(df)*100:.1f}%)")
        print(f"    Complete after    : {new_complete}/{len(df_enhanced)} ({new_complete/len(df_enhanced)*100:.1f}%)")
    
    # Save enhanced dataset
    print(f"\n{'='*80}")
    print(f"\nSaving enhanced dataset to {OUTPUT_CSV}...")
    df_enhanced.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved {len(df_enhanced)} rows")
    
    print(f"\nReport saved to {REPORT_FILE}")
    print(f"\n" + "="*80)
    print(f"ENHANCEMENT COMPLETE")
    print(f"="*80 + "\n")


if __name__ == "__main__":
    main()
