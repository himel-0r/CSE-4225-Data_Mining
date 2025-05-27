import time
import psutil
import os
from collections import defaultdict, Counter
from itertools import combinations, product
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Set, Tuple, Dict, Any

class SequencePatternMiner:
    """Base class for sequence pattern mining algorithms"""
    
    def __init__(self):
        self.sequences = []
        self.min_support = 0
        self.frequent_patterns = []
        
    def load_data(self, filename: str) -> List[List[List[int]]]:
        """
        Load and parse sequential transaction data from file
        Each transaction is delimited by -2, each event by -1
        """
        sequences = []
        
        try:
            with open(filename, 'r') as file:
                for line in file:
                    # Remove whitespace and split by spaces
                    items = [int(x) for x in line.strip().split() if x.isdigit() or x == '-1' or x == '-2']
                    
                    # Parse sequence: events are separated by -1, transactions by -2
                    current_sequence = []
                    current_event = []
                    
                    for item in items:
                        if item == -1:  # End of event
                            if current_event:
                                current_sequence.append(current_event)
                                current_event = []
                        elif item == -2:  # End of transaction/sequence
                            if current_event:
                                current_sequence.append(current_event)
                            if current_sequence:
                                sequences.append(current_sequence)
                            current_sequence = []
                            current_event = []
                        else:  # Regular item
                            current_event.append(item)
                    
                    # Handle case where sequence doesn't end with -2
                    if current_event:
                        current_sequence.append(current_event)
                    if current_sequence:
                        sequences.append(current_sequence)
                        
        except FileNotFoundError:
            print(f"File {filename} not found. Creating sample data...")
            sequences = self._create_sample_data()
            
        self.sequences = sequences
        return sequences
    
    def _create_sample_data(self) -> List[List[List[int]]]:
        """Create sample data for testing when kosarak.txt is not available"""
        return [
            [[1, 2], [3], [4, 5]],
            [[1], [3, 4], [5]],
            [[1, 2], [3, 4]],
            [[2], [3], [4, 5]],
            [[1], [4], [5]],
            [[1, 2, 3], [4]],
            [[2, 3], [4, 5]],
            [[1], [2], [3, 4, 5]],
            [[1, 3], [4]],
            [[2], [4, 5]]
        ]
    
    def get_support(self, pattern: List[List[int]]) -> int:
        """Calculate support count for a given pattern"""
        count = 0
        for sequence in self.sequences:
            if self._is_subsequence(pattern, sequence):
                count += 1
        return count
    
    def _is_subsequence(self, pattern: List[List[int]], sequence: List[List[int]]) -> bool:
        """Check if pattern is a subsequence of sequence"""
        if not pattern:
            return True
        if not sequence:
            return False
        
        p_idx = 0  # Pattern index
        
        for s_idx in range(len(sequence)):
            if p_idx >= len(pattern):
                break
                
            # Check if current pattern event can be matched with current sequence event
            if self._event_contains(sequence[s_idx], pattern[p_idx]):
                p_idx += 1
        
        return p_idx == len(pattern)
    
    def _event_contains(self, seq_event: List[int], pattern_event: List[int]) -> bool:
        """Check if sequence event contains all items in pattern event"""
        return all(item in seq_event for item in pattern_event)


class GSP(SequencePatternMiner):
    """Generalized Sequential Pattern (GSP) Algorithm Implementation"""
    
    def __init__(self):
        super().__init__()
        self.candidate_patterns = []
    
    def mine_patterns(self, min_support_percent: float) -> List[Tuple[List[List[int]], int]]:
        """
        Mine frequent sequential patterns using GSP algorithm
        Returns list of (pattern, support_count) tuples
        """
        total_sequences = len(self.sequences)
        self.min_support = max(1, int(min_support_percent * total_sequences / 100))
        
        # Step 1: Find frequent 1-sequences
        frequent_patterns = []
        L1 = self._generate_frequent_1_sequences()
        frequent_patterns.extend(L1)
        
        k = 2
        Lk_minus_1 = L1
        
        # Step 2: Generate k-sequences until no more frequent patterns
        while Lk_minus_1:
            # Generate candidates
            candidates = self._generate_candidates(Lk_minus_1, k)
            
            # Prune candidates and find frequent k-sequences
            Lk = []
            for candidate in candidates:
                support = self.get_support(candidate)
                if support >= self.min_support:
                    Lk.append((candidate, support))
            
            frequent_patterns.extend(Lk)
            Lk_minus_1 = [pattern for pattern, _ in Lk]
            k += 1
        
        self.frequent_patterns = frequent_patterns
        return frequent_patterns
    
    def _generate_frequent_1_sequences(self) -> List[Tuple[List[List[int]], int]]:
        """Generate frequent 1-sequences (single items)"""
        item_counts = Counter()
        
        # Count all items
        for sequence in self.sequences:
            items_in_sequence = set()
            for event in sequence:
                for item in event:
                    items_in_sequence.add(item)
            for item in items_in_sequence:
                item_counts[item] += 1
        
        # Filter frequent items
        frequent_1_sequences = []
        for item, count in item_counts.items():
            if count >= self.min_support:
                frequent_1_sequences.append(([[item]], count))
        
        return frequent_1_sequences
    
    def _generate_candidates(self, frequent_patterns: List[List[List[int]]], k: int) -> List[List[List[int]]]:
        """Generate candidate k-sequences from frequent (k-1)-sequences"""
        candidates = []
        
        for i in range(len(frequent_patterns)):
            for j in range(len(frequent_patterns)):
                # Join two (k-1)-sequences
                seq1, seq2 = frequent_patterns[i], frequent_patterns[j]
                
                # Case 1: Extend with new event
                if seq1[:-1] == seq2[1:] or (len(seq1) == 1 and len(seq2) == 1):
                    new_candidate = seq1 + [seq2[-1]]
                    if new_candidate not in candidates:
                        candidates.append(new_candidate)
                
                # Case 2: Extend last event
                if len(seq1) == len(seq2) and seq1[:-1] == seq2[:-1]:
                    if len(seq1[-1]) == 1 and len(seq2[-1]) == 1:
                        new_event = sorted(seq1[-1] + seq2[-1])
                        new_candidate = seq1[:-1] + [new_event]
                        if new_candidate not in candidates:
                            candidates.append(new_candidate)
        
        return candidates


class PrefixSpan(SequencePatternMiner):
    """PrefixSpan Algorithm Implementation"""
    
    def __init__(self):
        super().__init__()
        self.projected_databases = {}
    
    def mine_patterns(self, min_support_percent: float) -> List[Tuple[List[List[int]], int]]:
        """
        Mine frequent sequential patterns using PrefixSpan algorithm
        Returns list of (pattern, support_count) tuples
        """
        total_sequences = len(self.sequences)
        self.min_support = max(1, int(min_support_percent * total_sequences / 100))
        
        self.frequent_patterns = []
        
        # Find frequent items
        frequent_items = self._find_frequent_items()
        
        # For each frequent item, build projected database and mine recursively
        for item in frequent_items:
            pattern = [[item]]
            support = frequent_items[item]
            self.frequent_patterns.append((pattern, support))
            
            # Create projected database
            projected_db = self._create_projected_database(pattern)
            
            # Recursive mining
            self._prefix_span_recursive(pattern, projected_db)
        
        return self.frequent_patterns
    
    def _find_frequent_items(self) -> Dict[int, int]:
        """Find all frequent 1-items"""
        item_counts = Counter()
        
        for sequence in self.sequences:
            items_in_sequence = set()
            for event in sequence:
                for item in event:
                    items_in_sequence.add(item)
            for item in items_in_sequence:
                item_counts[item] += 1
        
        # Filter frequent items
        frequent_items = {}
        for item, count in item_counts.items():
            if count >= self.min_support:
                frequent_items[item] = count
        
        return frequent_items
    
    def _create_projected_database(self, prefix: List[List[int]]) -> List[List[List[int]]]:
        """Create projected database for given prefix"""
        projected_db = []
        
        for sequence in self.sequences:
            projected_seq = self._project_sequence(sequence, prefix)
            if projected_seq:
                projected_db.append(projected_seq)
        
        return projected_db
    
    def _project_sequence(self, sequence: List[List[int]], prefix: List[List[int]]) -> List[List[int]]:
        """Project a sequence with respect to the given prefix"""
        if not self._is_subsequence(prefix, sequence):
            return []
        
        # Find the position where prefix ends in sequence
        prefix_end_pos = self._find_prefix_end_position(sequence, prefix)
        if prefix_end_pos == -1:
            return []
        
        # Return suffix after prefix
        event_idx, item_idx = prefix_end_pos
        
        projected_seq = []
        
        # Add remaining items from the event where prefix ends
        if item_idx < len(sequence[event_idx]) - 1:
            remaining_items = sequence[event_idx][item_idx + 1:]
            if remaining_items:
                projected_seq.append(remaining_items)
        
        # Add all subsequent events
        projected_seq.extend(sequence[event_idx + 1:])
        
        return projected_seq
    
    def _find_prefix_end_position(self, sequence: List[List[int]], prefix: List[List[int]]) -> Tuple[int, int]:
        """Find the position where prefix ends in sequence"""
        if not prefix:
            return (0, -1)
        
        p_event_idx = 0
        p_item_idx = 0
        
        for s_event_idx in range(len(sequence)):
            if p_event_idx >= len(prefix):
                break
            
            # Try to match current prefix event with current sequence event
            prefix_event = prefix[p_event_idx]
            sequence_event = sequence[s_event_idx]
            
            matched_items = 0
            last_matched_idx = -1
            
            for p_item in prefix_event:
                found = False
                for s_item_idx in range(last_matched_idx + 1, len(sequence_event)):
                    if sequence_event[s_item_idx] == p_item:
                        matched_items += 1
                        last_matched_idx = s_item_idx
                        found = True
                        break
                if not found:
                    break
            
            if matched_items == len(prefix_event):
                p_event_idx += 1
                if p_event_idx == len(prefix):
                    return (s_event_idx, last_matched_idx)
        
        return (-1, -1)
    
    def _prefix_span_recursive(self, prefix: List[List[int]], projected_db: List[List[List[int]]]):
        """Recursively mine patterns using PrefixSpan"""
        # Find frequent items in projected database
        frequent_items = self._find_frequent_items_in_projected_db(projected_db)
        
        for item, count in frequent_items.items():
            # Create new pattern by extending prefix
            new_pattern_seq = prefix + [[item]]  # i-extension (new event)
            new_pattern_event = prefix[:-1] + [prefix[-1] + [item]]  # s-extension (same event)
            
            # Check which extension is valid and frequent
            for new_pattern in [new_pattern_seq, new_pattern_event]:
                support = self.get_support(new_pattern)
                if support >= self.min_support:
                    self.frequent_patterns.append((new_pattern, support))
                    
                    # Create new projected database and recurse
                    new_projected_db = self._create_projected_database(new_pattern)
                    if new_projected_db:
                        self._prefix_span_recursive(new_pattern, new_projected_db)
    
    def _find_frequent_items_in_projected_db(self, projected_db: List[List[List[int]]]) -> Dict[int, int]:
        """Find frequent items in projected database"""
        item_counts = Counter()
        
        for sequence in projected_db:
            items_in_sequence = set()
            for event in sequence:
                for item in event:
                    items_in_sequence.add(item)
            for item in items_in_sequence:
                item_counts[item] += 1
        
        # Filter frequent items
        frequent_items = {}
        for item, count in item_counts.items():
            if count >= self.min_support:
                frequent_items[item] = count
        
        return frequent_items


class PerformanceAnalyzer:
    """Class to analyze and compare algorithm performance"""
    
    def __init__(self):
        self.results = {
            'GSP': {'times': [], 'memory': [], 'patterns': []},
            'PrefixSpan': {'times': [], 'memory': [], 'patterns': []}
        }
        self.support_thresholds = []
    
    def run_comparison(self, filename: str, support_thresholds: List[float]):
        """Run performance comparison between GSP and PrefixSpan"""
        self.support_thresholds = support_thresholds
        
        print("Loading data...")
        # Load data once
        gsp = GSP()
        prefixspan = PrefixSpan()
        
        gsp.load_data(filename)
        prefixspan.sequences = gsp.sequences  # Share the same data
        
        print(f"Loaded {len(gsp.sequences)} sequences")
        
        for threshold in support_thresholds:
            print(f"\n=== Running algorithms with {threshold}% minimum support ===")
            
            # Test GSP
            print("Running GSP...")
            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            gsp_patterns = gsp.mine_patterns(threshold)
            
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            gsp_time = end_time - start_time
            gsp_memory = end_memory - start_memory
            
            self.results['GSP']['times'].append(gsp_time)
            self.results['GSP']['memory'].append(max(0, gsp_memory))  # Ensure non-negative
            self.results['GSP']['patterns'].append(len(gsp_patterns))
            
            print(f"GSP: {gsp_time:.3f}s, {gsp_memory:.2f}MB, {len(gsp_patterns)} patterns")
            
            # Test PrefixSpan
            print("Running PrefixSpan...")
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            ps_patterns = prefixspan.mine_patterns(threshold)
            
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            ps_time = end_time - start_time
            ps_memory = end_memory - start_memory
            
            self.results['PrefixSpan']['times'].append(ps_time)
            self.results['PrefixSpan']['memory'].append(max(0, ps_memory))  # Ensure non-negative
            self.results['PrefixSpan']['patterns'].append(len(ps_patterns))
            
            print(f"PrefixSpan: {ps_time:.3f}s, {ps_memory:.2f}MB, {len(ps_patterns)} patterns")
    
    def plot_results(self):
        """Plot performance comparison results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Execution Time
        ax1.plot(self.support_thresholds, self.results['GSP']['times'], 
                'o-', label='GSP', linewidth=2, markersize=8)
        ax1.plot(self.support_thresholds, self.results['PrefixSpan']['times'], 
                's-', label='PrefixSpan', linewidth=2, markersize=8)
        ax1.set_xlabel('Minimum Support Percentage (%)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory Usage
        ax2.plot(self.support_thresholds, self.results['GSP']['memory'], 
                'o-', label='GSP', linewidth=2, markersize=8)
        ax2.plot(self.support_thresholds, self.results['PrefixSpan']['memory'], 
                's-', label='PrefixSpan', linewidth=2, markersize=8)
        ax2.set_xlabel('Minimum Support Percentage (%)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Number of Patterns Found
        ax3.plot(self.support_thresholds, self.results['GSP']['patterns'], 
                'o-', label='GSP', linewidth=2, markersize=8)
        ax3.plot(self.support_thresholds, self.results['PrefixSpan']['patterns'], 
                's-', label='PrefixSpan', linewidth=2, markersize=8)
        ax3.set_xlabel('Minimum Support Percentage (%)')
        ax3.set_ylabel('Number of Patterns Found')
        ax3.set_title('Patterns Found Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Combined Performance Score (lower is better)
        # Normalize time and memory to same scale and combine
        gsp_norm_time = np.array(self.results['GSP']['times']) / max(max(self.results['GSP']['times']), 1e-6)
        gsp_norm_memory = np.array(self.results['GSP']['memory']) / max(max(self.results['GSP']['memory']), 1e-6)
        gsp_score = gsp_norm_time + gsp_norm_memory
        
        ps_norm_time = np.array(self.results['PrefixSpan']['times']) / max(max(self.results['PrefixSpan']['times']), 1e-6)
        ps_norm_memory = np.array(self.results['PrefixSpan']['memory']) / max(max(self.results['PrefixSpan']['memory']), 1e-6)
        ps_score = ps_norm_time + ps_norm_memory
        
        ax4.plot(self.support_thresholds, gsp_score, 'o-', label='GSP', linewidth=2, markersize=8)
        ax4.plot(self.support_thresholds, ps_score, 's-', label='PrefixSpan', linewidth=2, markersize=8)
        ax4.set_xlabel('Minimum Support Percentage (%)')
        ax4.set_ylabel('Combined Performance Score (lower is better)')
        ax4.set_title('Overall Performance Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sequence_mining_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print performance summary and analysis"""
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nSupport Thresholds Tested: {self.support_thresholds}")
        
        print("\n--- EXECUTION TIME ANALYSIS ---")
        avg_gsp_time = np.mean(self.results['GSP']['times'])
        avg_ps_time = np.mean(self.results['PrefixSpan']['times'])
        print(f"Average GSP Time: {avg_gsp_time:.4f} seconds")
        print(f"Average PrefixSpan Time: {avg_ps_time:.4f} seconds")
        
        if avg_gsp_time < avg_ps_time:
            speedup = avg_ps_time / avg_gsp_time
            print(f"GSP is {speedup:.2f}x faster on average")
        else:
            speedup = avg_gsp_time / avg_ps_time
            print(f"PrefixSpan is {speedup:.2f}x faster on average")
        
        print("\n--- MEMORY USAGE ANALYSIS ---")
        avg_gsp_memory = np.mean(self.results['GSP']['memory'])
        avg_ps_memory = np.mean(self.results['PrefixSpan']['memory'])
        print(f"Average GSP Memory: {avg_gsp_memory:.2f} MB")
        print(f"Average PrefixSpan Memory: {avg_ps_memory:.2f} MB")
        
        print("\n--- PATTERNS FOUND ANALYSIS ---")
        print("Support% | GSP Patterns | PrefixSpan Patterns | Difference")
        print("-" * 60)
        for i, threshold in enumerate(self.support_thresholds):
            gsp_patterns = self.results['GSP']['patterns'][i]
            ps_patterns = self.results['PrefixSpan']['patterns'][i]
            diff = abs(gsp_patterns - ps_patterns)
            print(f"{threshold:7.1f} | {gsp_patterns:11d} | {ps_patterns:16d} | {diff:10d}")
        
        print("\n--- ALGORITHM RECOMMENDATIONS ---")
        if avg_gsp_time < avg_ps_time and avg_gsp_memory < avg_ps_memory:
            print("✓ GSP performs better in both time and memory efficiency")
        elif avg_ps_time < avg_gsp_time and avg_ps_memory < avg_gsp_memory:
            print("✓ PrefixSpan performs better in both time and memory efficiency")
        else:
            print("✓ Performance trade-offs exist between algorithms:")
            if avg_gsp_time < avg_ps_time:
                print("  - GSP is faster but uses more memory")
            else:
                print("  - PrefixSpan is faster but GSP uses less memory")
        
        print("\n--- KEY INSIGHTS ---")
        print("• Higher support thresholds typically result in:")
        print("  - Faster execution (fewer patterns to explore)")
        print("  - Lower memory usage (smaller search space)")
        print("  - Fewer patterns found (more restrictive threshold)")
        
        print("• Algorithm characteristics:")
        print("  - GSP: Breadth-first approach, generates candidates explicitly")
        print("  - PrefixSpan: Depth-first approach, uses projected databases")
        print("  - Choice depends on data characteristics and requirements")


def main():
    """Main function to run the sequence pattern mining comparison"""
    print("Sequence Pattern Mining: GSP vs PrefixSpan Comparison")
    print("=" * 60)
    
    # Configuration
    filename = 'book.txt'
    support_thresholds = [50.0, 75.0]  # Percentage values
    
    # Create analyzer and run comparison
    analyzer = PerformanceAnalyzer()
    
    # try:
    analyzer.run_comparison(filename, support_thresholds)
        # analyzer.plot_results()
        # analyzer.print_summary()
        
    # except Exception as e:
    #     print(f"Error during execution: {e}")
    #     print("Note: If kosarak.txt is not available, the program uses sample data for demonstration.")


if __name__ == "__main__":
    main()