"""
ONTOLOGY BATCH MERGER
Merges multiple JSONL batch files from Grok's ontological mapping
into a unified structure compatible with the Dendritic Surgery system.

Usage:
    python ontology_merger.py --input-dir ./batches --output merged_ontology.json
    python ontology_merger.py --files batch_001.jsonl batch_002.jsonl --output merged.json
    python ontology_merger.py --input-dir ./batches --stats-only

Author: Tom's AI Roundtable (Claude contribution)
Target: Python 3.10+
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class MergeConflict:
    """Records when same lemma appears with different data"""
    lemma: str
    field: str
    values: list
    resolution: str
    sources: list[str]


@dataclass 
class MergeStats:
    """Statistics from the merge operation"""
    total_files: int = 0
    total_entries: int = 0
    unique_lemmas: int = 0
    duplicates_found: int = 0
    conflicts_resolved: int = 0
    parent_distribution: dict = field(default_factory=dict)
    pos_distribution: dict = field(default_factory=dict)
    domain_distribution: dict = field(default_factory=dict)
    confidence_histogram: dict = field(default_factory=dict)
    flags_frequency: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "total_entries": self.total_entries,
            "unique_lemmas": self.unique_lemmas,
            "duplicates_found": self.duplicates_found,
            "conflicts_resolved": self.conflicts_resolved,
            "parent_distribution": dict(sorted(
                self.parent_distribution.items(), 
                key=lambda x: -x[1]
            )),
            "pos_distribution": dict(sorted(
                self.pos_distribution.items(),
                key=lambda x: -x[1]
            )),
            "domain_distribution": dict(sorted(
                self.domain_distribution.items(),
                key=lambda x: -x[1]
            )),
            "confidence_histogram": self.confidence_histogram,
            "top_50_flags": dict(sorted(
                self.flags_frequency.items(),
                key=lambda x: -x[1]
            )[:50])
        }


class ConflictResolution:
    """Strategies for resolving merge conflicts"""
    
    @staticmethod
    def higher_confidence(entries: list[dict]) -> dict:
        """Pick entry with highest confidence"""
        return max(entries, key=lambda e: e.get("confidence", 0))
    
    @staticmethod
    def merge_flags(entries: list[dict]) -> list:
        """Union of all flags from all entries"""
        all_flags = set()
        for entry in entries:
            all_flags.update(entry.get("flags", []))
        return sorted(list(all_flags))
    
    @staticmethod
    def first_seen(entries: list[dict]) -> dict:
        """Keep first occurrence"""
        return entries[0]
    
    @staticmethod
    def merge_all_senses(entries: list[dict]) -> dict:
        """Merge entries, combining flags and keeping highest confidence"""
        base = ConflictResolution.higher_confidence(entries)
        merged = base.copy()
        merged["flags"] = ConflictResolution.merge_flags(entries)
        merged["_merge_sources"] = len(entries)
        return merged


class OntologyBatchMerger:
    """
    Merges multiple JSONL ontology batch files into unified structure.
    
    Output format (Dendritic Surgery compatible):
    {
        "metadata": {...},
        "parents": {"Entity": null, "PhysicalObject": "Entity", ...},
        "children": {"Entity": ["PhysicalObject", "AbstractEntity"], ...},
        "entries": {"lemma": {...}, ...}
    }
    """
    
    # Known top-level parents in Grok's schema
    ROOT_PARENTS = {
        "Entity", "PhysicalObject", "AbstractEntity", 
        "InformationalEntity", "TemporalEntity", "Location",
        "Event", "Process", "Quality", "Relation"
    }
    
    def __init__(self, conflict_strategy: str = "merge_all"):
        self.entries: dict[str, dict] = {}
        self.conflicts: list[MergeConflict] = []
        self.stats = MergeStats()
        self.source_files: list[str] = []
        
        # Select conflict resolution strategy
        self.conflict_resolver = {
            "higher_confidence": ConflictResolution.higher_confidence,
            "first_seen": ConflictResolution.first_seen,
            "merge_all": ConflictResolution.merge_all_senses
        }.get(conflict_strategy, ConflictResolution.merge_all_senses)
    
    def load_jsonl(self, filepath: Path) -> list[dict]:
        """Load entries from a JSONL file"""
        entries = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry["_source_file"] = filepath.name
                    entry["_source_line"] = line_num
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"  ⚠ JSON error in {filepath.name}:{line_num}: {e}")
        return entries
    
    def add_batch(self, filepath: Path) -> int:
        """Add entries from a batch file, handling duplicates"""
        entries = self.load_jsonl(filepath)
        self.source_files.append(filepath.name)
        self.stats.total_files += 1
        added = 0
        
        for entry in entries:
            self.stats.total_entries += 1
            lemma = entry.get("lemma", "").strip().lower()
            
            if not lemma:
                continue
            
            if lemma in self.entries:
                # Duplicate found - resolve conflict
                self.stats.duplicates_found += 1
                existing = self.entries[lemma]
                
                # Check if actually different
                if self._entries_differ(existing, entry):
                    self.stats.conflicts_resolved += 1
                    resolved = self.conflict_resolver([existing, entry])
                    
                    # Record the conflict
                    self.conflicts.append(MergeConflict(
                        lemma=lemma,
                        field="multiple",
                        values=[existing.get("parent"), entry.get("parent")],
                        resolution=self.conflict_resolver.__name__,
                        sources=[existing.get("_source_file"), entry.get("_source_file")]
                    ))
                    
                    self.entries[lemma] = resolved
                else:
                    # Identical duplicate - just merge flags
                    existing["flags"] = ConflictResolution.merge_flags([existing, entry])
            else:
                self.entries[lemma] = entry
                added += 1
        
        return added
    
    def _entries_differ(self, e1: dict, e2: dict) -> bool:
        """Check if two entries have meaningful differences"""
        key_fields = ["parent", "kind", "pos_guess", "domain"]
        for field in key_fields:
            if e1.get(field) != e2.get(field):
                return True
        return False
    
    def load_directory(self, dirpath: Path, pattern: str = "batch_*.jsonl"):
        """Load all matching files from directory"""
        files = sorted(dirpath.glob(pattern))
        print(f"\n📂 Found {len(files)} batch files matching '{pattern}'")
        
        for filepath in files:
            added = self.add_batch(filepath)
            print(f"  ✓ {filepath.name}: {added} new entries")
        
        print(f"\n📊 Loaded {self.stats.total_entries} total entries")
        print(f"   → {len(self.entries)} unique lemmas")
        print(f"   → {self.stats.duplicates_found} duplicates handled")
    
    def build_graph_structure(self) -> tuple[dict, dict]:
        """Build parent->child graph from entries"""
        parents = {}  # node -> parent
        children = defaultdict(list)  # node -> [children]
        
        # First, establish root parents
        for root in self.ROOT_PARENTS:
            parents[root] = "Entity" if root != "Entity" else None
            if root != "Entity":
                children["Entity"].append(root)
        
        # Then add all entries
        for lemma, entry in self.entries.items():
            parent = entry.get("parent")
            if parent:
                parents[lemma] = parent
                children[parent].append(lemma)
        
        # Sort children lists for consistency
        for parent in children:
            children[parent] = sorted(set(children[parent]))
        
        return parents, dict(children)
    
    def compute_statistics(self):
        """Compute comprehensive statistics about the merged ontology"""
        for lemma, entry in self.entries.items():
            # Parent distribution
            parent = entry.get("parent", "Unknown")
            self.stats.parent_distribution[parent] = \
                self.stats.parent_distribution.get(parent, 0) + 1
            
            # POS distribution
            pos = entry.get("pos_guess", "Unknown")
            self.stats.pos_distribution[pos] = \
                self.stats.pos_distribution.get(pos, 0) + 1
            
            # Domain distribution
            domain = entry.get("domain") or "unspecified"
            self.stats.domain_distribution[domain] = \
                self.stats.domain_distribution.get(domain, 0) + 1
            
            # Confidence histogram (buckets of 0.1)
            conf = entry.get("confidence", 0)
            bucket = f"{int(conf * 10) / 10:.1f}-{int(conf * 10) / 10 + 0.09:.2f}"
            self.stats.confidence_histogram[bucket] = \
                self.stats.confidence_histogram.get(bucket, 0) + 1
            
            # Flag frequency
            for flag in entry.get("flags", []):
                self.stats.flags_frequency[flag] = \
                    self.stats.flags_frequency.get(flag, 0) + 1
        
        self.stats.unique_lemmas = len(self.entries)
    
    def export(self, output_path: Path, include_stats: bool = True) -> dict:
        """Export merged ontology to JSON file"""
        self.compute_statistics()
        parents, children = self.build_graph_structure()
        
        # Clean entries for export (remove internal fields)
        clean_entries = {}
        for lemma, entry in self.entries.items():
            clean = {k: v for k, v in entry.items() 
                    if not k.startswith("_")}
            clean_entries[lemma] = clean
        
        output = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "source_files": self.source_files,
                "merger_version": "1.0.0",
                "total_entries": len(clean_entries),
                "conflict_strategy": self.conflict_resolver.__name__
            },
            "parents": parents,
            "children": children,
            "entries": clean_entries
        }
        
        if include_stats:
            output["statistics"] = self.stats.to_dict()
        
        if self.conflicts:
            output["conflicts_log"] = [
                {
                    "lemma": c.lemma,
                    "field": c.field,
                    "values": c.values,
                    "resolution": c.resolution,
                    "sources": c.sources
                }
                for c in self.conflicts[:100]  # Cap at 100 for readability
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return output
    
    def print_report(self):
        """Print human-readable merge report"""
        self.compute_statistics()
        stats = self.stats
        
        print("\n" + "=" * 60)
        print("🧠 ONTOLOGY MERGE REPORT")
        print("=" * 60)
        
        print(f"\n📁 Source Files: {stats.total_files}")
        print(f"📝 Total Entries Processed: {stats.total_entries}")
        print(f"🔤 Unique Lemmas: {stats.unique_lemmas}")
        print(f"🔄 Duplicates Found: {stats.duplicates_found}")
        print(f"⚔️  Conflicts Resolved: {stats.conflicts_resolved}")
        
        print("\n📊 PARENT DISTRIBUTION (Top 10):")
        print("-" * 40)
        for parent, count in list(stats.parent_distribution.items())[:10]:
            pct = count / stats.unique_lemmas * 100
            bar = "█" * int(pct / 2)
            print(f"  {parent:25} {count:5} ({pct:5.1f}%) {bar}")
        
        print("\n🏷️  POS DISTRIBUTION:")
        print("-" * 40)
        for pos, count in sorted(stats.pos_distribution.items(), 
                                 key=lambda x: -x[1])[:10]:
            pct = count / stats.unique_lemmas * 100
            print(f"  {pos:25} {count:5} ({pct:5.1f}%)")
        
        print("\n🌐 DOMAIN DISTRIBUTION (Top 10):")
        print("-" * 40)
        for domain, count in list(stats.domain_distribution.items())[:10]:
            pct = count / stats.unique_lemmas * 100
            print(f"  {domain:25} {count:5} ({pct:5.1f}%)")
        
        print("\n🎯 CONFIDENCE DISTRIBUTION:")
        print("-" * 40)
        for bucket in sorted(stats.confidence_histogram.keys()):
            count = stats.confidence_histogram[bucket]
            pct = count / stats.unique_lemmas * 100
            bar = "█" * int(pct / 2)
            print(f"  {bucket:12} {count:5} ({pct:5.1f}%) {bar}")
        
        print("\n🚩 TOP 20 FLAGS:")
        print("-" * 40)
        for flag, count in list(stats.flags_frequency.items())[:20]:
            print(f"  {flag:25} {count:5}")
        
        if self.conflicts:
            print(f"\n⚠️  SAMPLE CONFLICTS (showing 5 of {len(self.conflicts)}):")
            print("-" * 40)
            for conflict in self.conflicts[:5]:
                print(f"  '{conflict.lemma}': {conflict.values}")
                print(f"    Sources: {conflict.sources}")
                print(f"    Resolution: {conflict.resolution}")
        
        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL ontology batch files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir ./batches --output merged.json
  %(prog)s --files batch_001.jsonl batch_002.jsonl -o merged.json
  %(prog)s --input-dir ./batches --stats-only
  %(prog)s --input-dir ./batches --conflict-strategy higher_confidence
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir", "-d",
        type=Path,
        help="Directory containing batch JSONL files"
    )
    input_group.add_argument(
        "--files", "-f",
        nargs="+",
        type=Path,
        help="Specific JSONL files to merge"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("merged_ontology.json"),
        help="Output JSON file path (default: merged_ontology.json)"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        default="*.jsonl",
        help="Glob pattern for batch files (default: *.jsonl)"
    )
    
    parser.add_argument(
        "--conflict-strategy", "-c",
        choices=["merge_all", "higher_confidence", "first_seen"],
        default="merge_all",
        help="Strategy for resolving duplicate entries (default: merge_all)"
    )
    
    parser.add_argument(
        "--stats-only", "-s",
        action="store_true",
        help="Only print statistics, don't write output file"
    )
    
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Exclude statistics from output file"
    )
    
    args = parser.parse_args()
    
    # Initialize merger
    merger = OntologyBatchMerger(conflict_strategy=args.conflict_strategy)
    
    # Load files
    if args.input_dir:
        if not args.input_dir.exists():
            print(f"❌ Directory not found: {args.input_dir}")
            return 1
        merger.load_directory(args.input_dir, args.pattern)
    else:
        for filepath in args.files:
            if not filepath.exists():
                print(f"❌ File not found: {filepath}")
                return 1
            added = merger.add_batch(filepath)
            print(f"✓ {filepath.name}: {added} new entries")
    
    # Print report
    merger.print_report()
    
    # Export if not stats-only
    if not args.stats_only:
        result = merger.export(args.output, include_stats=not args.no_stats)
        print(f"\n✅ Merged ontology written to: {args.output}")
        print(f"   Total size: {args.output.stat().st_size / 1024:.1f} KB")
    
    return 0


if __name__ == "__main__":
    exit(main())