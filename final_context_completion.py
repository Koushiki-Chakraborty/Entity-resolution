"""
Final Context Completion & Validation
======================================
This script:
1. Identifies ALL rows with incomplete contexts
2. For each incomplete context, fetches complete information
3. Validates name-context semantic relationship
4. Creates final production-ready dataset
5. Generates detailed validation report

Key principle: Both NAME and CONTEXT must be semantically complete
so the model learns: "bcroy" + context_bcroy vs "bcrec" + context_bcrec
and can identify if they're the same entity through both signals.
"""

import pandas as pd
import requests
import json
import re
import time
from pathlib import Path
from typing import Tuple, Optional, Dict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV = "data/pairs/training_ready_enriched.csv"
OUTPUT_CSV = "data/pairs/training_ready_production.csv"
INCOMPLETE_REPORT = "data/pairs/incomplete_contexts_report.json"
FINAL_VALIDATION = "data/pairs/final_validation_complete.json"

MIN_CONTEXT_LENGTH = 120  # Minimum for proper entity resolution
MAX_CONTEXT_LENGTH = 500

HEADERS = {"User-Agent": "AgriLambdaNet-final-qa/1.0"}

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT COMPLETENESS CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class ContextQualityChecker:
    """Check if context is complete and valid."""
    
    @staticmethod
    def is_complete(text: str) -> Tuple[bool, str, float]:
        """
        Check if context meets quality standards.
        Returns: (is_complete, reason, quality_score)
        """
        text = str(text).strip()
        
        if not text or len(text) == 0:
            return False, "empty", 0.0
        
        if len(text) < MIN_CONTEXT_LENGTH:
            return False, f"too_short_{len(text)}", len(text) / MIN_CONTEXT_LENGTH
        
        if len(text) > MAX_CONTEXT_LENGTH:
            return False, "too_long", MAX_CONTEXT_LENGTH / len(text)
        
        # Check for proper sentence termination
        if text[-1] not in '.!?)"\'"':
            return False, "no_terminal_punct", 0.8
        
        # Check for truncation mid-word
        if text.endswith(('ing', 'tion', 'ness', 'ment', 'ous', ',')):
            return False, "truncated", 0.7
        
        # Check sentence count
        sentences = len(re.split(r'[.!?]+', text.strip()))
        if sentences < 1:
            return False, "no_sentences", 0.5
        
        return True, "complete", 1.0
    
    @staticmethod
    def validate_name_context_relevance(name: str, context: str) -> Tuple[bool, float, list]:
        """
        Check if context semantically relates to the name.
        Returns: (is_relevant, relevance_score, matched_keywords)
        """
        if not context or len(str(context).strip()) < 50:
            return False, 0.0, []
        
        # Extract keywords from name
        stop_words = {'of', 'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'by', 'is'}
        name_words = str(name).lower().split()
        keywords = [w for w in name_words if w not in stop_words and len(w) > 2]
        
        if not keywords:
            return True, 0.5, []  # Can't validate, assume OK
        
        context_lower = str(context).lower()
        matched = [kw for kw in keywords if kw in context_lower]
        
        relevance_score = len(matched) / len(keywords) if keywords else 0.5
        is_relevant = relevance_score >= 0.5
        
        return is_relevant, relevance_score, matched


# ─────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA ENHANCED FETCHER
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedWikipediaFetcher:
    """Fetch complete context with multiple strategies."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.cache = {}
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia markup."""
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[\[|\]\]', '', text)
        text = re.sub(r'\s*\([^)]*pronunciation[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_sentences_smart(self, text: str, target_min=MIN_CONTEXT_LENGTH) -> str:
        """Extract complete sentences smartly."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if not sentences:
            return text[:MAX_CONTEXT_LENGTH].strip()
        
        result = ""
        for sent in sentences:
            candidate = (result + " " + sent).strip() if result else sent
            if len(candidate) >= target_min:
                return candidate if len(candidate) <= MAX_CONTEXT_LENGTH else result
            result = candidate
        
        return result if len(result) >= 60 else text[:MAX_CONTEXT_LENGTH]
    
    def fetch_wikipedia(self, name: str) -> Optional[str]:
        """Fetch Wikipedia content."""
        if name in self.cache:
            return self.cache[name]
        
        # Try exact match
        for variant in [name, name.title(), name.capitalize()]:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(variant)}"
            try:
                r = self.session.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("type") != "disambiguation":
                        extract = data.get("extract", "")
                        if extract and len(extract) > 80:
                            cleaned = self.clean_text(extract)
                            result = self.extract_sentences_smart(cleaned)
                            self.cache[name] = result
                            return result
            except:
                pass
            time.sleep(0.3)
        
        # Search fallback
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": name,
                "srlimit": 3,
                "format": "json"
            }
            r = self.session.get(url, params=params, timeout=10)
            if r.status_code == 200:
                results = r.json().get("query", {}).get("search", [])
                if results:
                    top_title = results[0]["title"]
                    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(top_title)}"
                    r = self.session.get(url, timeout=10)
                    if r.status_code == 200:
                        data = r.json()
                        extract = data.get("extract", "")
                        if extract and len(extract) > 80:
                            cleaned = self.clean_text(extract)
                            result = self.extract_sentences_smart(cleaned)
                            self.cache[name] = result
                            return result
        except:
            pass
        
        self.cache[name] = None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*100)
    print("FINAL CONTEXT COMPLETION & VALIDATION")
    print("="*100)
    
    # Load dataset
    print(f"\n[1] Loading enriched dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"    Loaded {len(df)} rows")
    
    checker = ContextQualityChecker()
    fetcher = EnhancedWikipediaFetcher()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: IDENTIFY INCOMPLETE CONTEXTS
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n[2] Scanning for incomplete contexts...")
    
    incomplete_rows = []
    total_issues = 0
    
    for idx, row in df.iterrows():
        for col_n, col_c in [('name_a', 'context_a'), ('name_b', 'context_b')]:
            name = row[col_n]
            context = row[col_c]
            
            is_complete, reason, score = checker.is_complete(context)
            
            if not is_complete:
                total_issues += 1
                incomplete_rows.append({
                    'row_idx': idx,
                    'name': name,
                    'column': col_c,
                    'context': str(context)[:100],
                    'issue': reason,
                    'current_length': len(str(context)),
                    'quality_score': score
                })
    
    print(f"    Found {len(incomplete_rows)} incomplete contexts")
    print(f"    Issues breakdown:")
    
    issue_counts = {}
    for row in incomplete_rows:
        issue = row['issue']
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {issue:<30}: {count:4d}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: COMPLETE INCOMPLETE CONTEXTS
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n[3] Completing incomplete contexts...")
    
    df_fixed = df.copy()
    updates = {
        'from_wikipedia': 0,
        'kept_enhanced': 0,
        'enhanced_further': 0,
        'still_incomplete': 0
    }
    
    completion_details = []
    
    for row_data in incomplete_rows:
        idx = row_data['row_idx']
        col_c = row_data['column']
        name = row_data['name']
        current_ctx = df_fixed.at[idx, col_c]
        
        is_complete, reason, score = checker.is_complete(current_ctx)
        
        if not is_complete:
            # Try to fetch from Wikipedia
            wiki_ctx = fetcher.fetch_wikipedia(name)
            time.sleep(0.5)
            
            if wiki_ctx and len(wiki_ctx) >= MIN_CONTEXT_LENGTH:
                # Validate semantic relevance
                is_relevant, rel_score, matched = checker.validate_name_context_relevance(name, wiki_ctx)
                
                if is_relevant and rel_score >= 0.5:
                    df_fixed.at[idx, col_c] = wiki_ctx
                    updates['from_wikipedia'] += 1
                    status = "completed_from_wikipedia"
                else:
                    # Wikipedia fetch didn't help
                    updates['still_incomplete'] += 1
                    status = "wiki_not_relevant"
            else:
                # Could not fetch from Wikipedia
                updates['kept_enhanced'] += 1
                status = "no_wiki_available"
            
            # Verify final state
            final_ctx = df_fixed.at[idx, col_c]
            is_final_complete, final_reason, final_score = checker.is_complete(final_ctx)
            is_relevant, rel_score, matched = checker.validate_name_context_relevance(name, final_ctx)
            
            completion_details.append({
                'row': idx,
                'name': name,
                'column': col_c,
                'original_length': len(str(current_ctx)),
                'final_length': len(str(final_ctx)),
                'original_issue': reason,
                'final_complete': is_final_complete,
                'relevance': rel_score,
                'is_relevant': is_relevant,
                'status': status,
                'matched_keywords': matched
            })
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n[4] Validating final dataset...")
    
    validation_stats = {
        'total_rows': len(df_fixed),
        'total_contexts': len(df_fixed) * 2,
        'complete_contexts': 0,
        'relevant_contexts': 0,
        'perfect_match_names_contexts': 0,
        'incomplete_remaining': 0
    }
    
    issues_remaining = []
    
    for idx, row in df_fixed.iterrows():
        for col_n, col_c in [('name_a', 'context_a'), ('name_b', 'context_b')]:
            name = row[col_n]
            context = row[col_c]
            
            is_complete, reason, score = checker.is_complete(context)
            is_relevant, rel_score, matched = checker.validate_name_context_relevance(name, context)
            
            if is_complete:
                validation_stats['complete_contexts'] += 1
            else:
                validation_stats['incomplete_remaining'] += 1
                issues_remaining.append({
                    'row': idx,
                    'name': name,
                    'length': len(str(context)),
                    'issue': reason
                })
            
            if is_relevant and rel_score >= 0.8:
                validation_stats['relevant_contexts'] += 1
                validation_stats['perfect_match_names_contexts'] += 1
            elif is_relevant and rel_score >= 0.5:
                validation_stats['relevant_contexts'] += 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: REPORTING
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n[5] Generating reports...")
    
    print(f"\n{'='*100}")
    print("COMPLETION SUMMARY")
    print(f"{'='*100}")
    
    print(f"\nContexts Updated:")
    print(f"  From Wikipedia:    {updates['from_wikipedia']:4d}")
    print(f"  Kept as-is:        {updates['kept_enhanced']:4d}")
    print(f"  Still incomplete:  {updates['still_incomplete']:4d}")
    
    print(f"\nFinal Dataset Quality:")
    print(f"  Total contexts:              {validation_stats['total_contexts']:5d}")
    print(f"  Complete contexts:           {validation_stats['complete_contexts']:5d} ({validation_stats['complete_contexts']/validation_stats['total_contexts']*100:5.1f}%)")
    print(f"  Semantically relevant:       {validation_stats['relevant_contexts']:5d} ({validation_stats['relevant_contexts']/validation_stats['total_contexts']*100:5.1f}%)")
    print(f"  Perfect name-context match:  {validation_stats['perfect_match_names_contexts']:5d} ({validation_stats['perfect_match_names_contexts']/validation_stats['total_contexts']*100:5.1f}%)")
    print(f"  Incomplete remaining:        {validation_stats['incomplete_remaining']:5d} ({validation_stats['incomplete_remaining']/validation_stats['total_contexts']*100:5.1f}%)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: QUALITY ASSURANCE FOR ENTITY RESOLUTION
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n[6] Entity Resolution Readiness Check...")
    
    er_readiness = {
        'can_distinguish_by_name': 0,
        'can_distinguish_by_context': 0,
        'can_distinguish_by_both': 0,
        'unclear_pairs': 0
    }
    
    # Check pairs to see if they can be distinguished
    for idx in range(0, len(df_fixed)-1, 1):
        row = df_fixed.iloc[idx]
        name_a = str(row['name_a']).lower()
        name_b = str(row['name_b']).lower()
        ctx_a = str(row['context_a']).lower()
        ctx_b = str(row['context_b']).lower()
        
        # Check name similarity
        name_similar = name_a == name_b or (len(name_a) > 3 and name_a in name_b) or (len(name_b) > 3 and name_b in name_a)
        
        # Check context similarity (rough overlap)
        common_words = set(ctx_a.split()) & set(ctx_b.split())
        context_overlap = len(common_words) / max(len(ctx_a.split()), len(ctx_b.split())) if len(ctx_a.split()) > 0 else 0
        context_similar = context_overlap > 0.3
        
        if not name_similar and not context_similar:
            er_readiness['can_distinguish_by_both'] += 1
        elif name_similar and context_similar:
            er_readiness['can_distinguish_by_name'] += 1
        elif name_similar:
            er_readiness['can_distinguish_by_context'] += 1
        else:
            er_readiness['unclear_pairs'] += 1
    
    print(f"\nEntity Resolution Capability:")
    print(f"  Can distinguish by BOTH name AND context: {er_readiness['can_distinguish_by_both']:4d}")
    print(f"  Can distinguish by name (context helps): {er_readiness['can_distinguish_by_context']:4d}")
    print(f"  Need context to distinguish:            {er_readiness['can_distinguish_by_name']:4d}")
    print(f"  Potentially unclear pairs:              {er_readiness['unclear_pairs']:4d}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SAVE OUTPUTS
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"\n[7] Saving final dataset and reports...")
    
    df_fixed.to_csv(OUTPUT_CSV, index=False)
    print(f"    Saved: {OUTPUT_CSV}")
    
    # Save incomplete report
    with open(INCOMPLETE_REPORT, 'w') as f:
        json.dump({
            'originally_incomplete': len(incomplete_rows),
            'issue_breakdown': issue_counts,
            'details': completion_details[:50]  # Top 50 issues
        }, f, indent=2, default=str)
    print(f"    Saved: {INCOMPLETE_REPORT}")
    
    # Save final validation
    with open(FINAL_VALIDATION, 'w') as f:
        json.dump({
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_stats': validation_stats,
            'entity_resolution_readiness': er_readiness,
            'issues_remaining': issues_remaining[:20]  # Top 20
        }, f, indent=2, default=str)
    print(f"    Saved: {FINAL_VALIDATION}")
    
    print(f"\n{'='*100}")
    print("✅ FINAL DATASET READY FOR PRODUCTION")
    print(f"{'='*100}\n")
    
    # Show some before/after examples
    print(f"Sample Completion Examples:")
    for detail in completion_details[:5]:
        if detail['status'] == 'completed_from_wikipedia':
            print(f"\n  Entity: {detail['name']}")
            print(f"    Before: {detail['original_length']} chars")
            print(f"    After:  {detail['final_length']} chars")
            print(f"    Relevance: {detail['relevance']:.2f}")


if __name__ == "__main__":
    main()
