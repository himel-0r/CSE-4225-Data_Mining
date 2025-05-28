import sys
import os
import argparse
from typing import List, Dict, Tuple, Set, Any
import time
from collections import defaultdict

# Add parent directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

class PrefixSpan:
    def __init__(self, min_support_ratio: float = 0.1, verbose: bool = False):
        """
        Initialize PrefixSpan algorithm.
        
        Args:
            min_support_ratio (float): Minimum support ratio (0 to 1)
            verbose (bool): Whether to print detailed information about patterns
        """
        self.min_support_ratio = min_support_ratio
        self.verbose = verbose
        self.patterns: Dict[int, List[Tuple[List[List[int]], int]]] = {}  # Length -> [(pattern, support)]
        self.total_sequences = 0
        self.min_support = 0
    
    def _find_frequent_items(self, projected_db: List[Tuple[List[List[int]], int]]) -> Dict[int, int]:
        """
        Find all frequent items in the projected database.
        
        Args:
            projected_db: List of (sequence, position) pairs
            
        Returns:
            Dict mapping item to its support count
        """
        item_count: Dict[int, int] = defaultdict(int)
        
        # Count frequency of each item
        for sequence, pos in projected_db:
            # Get the remaining sequence after the prefix
            remaining = sequence[pos:]
            if not remaining:
                continue
                
            # Keep track of items we've seen in this sequence to avoid counting duplicates
            seen_items: Set[int] = set()
            
            # Count items in remaining sequence
            for itemset in remaining:
                for item in itemset:
                    if item not in seen_items:
                        item_count[item] += 1
                        seen_items.add(item)
        
        # Filter items that meet minimum support
        frequent_items = {item: count for item, count in item_count.items() 
                         if count >= self.min_support}
        
        return frequent_items
    
    def _project_database(self, projected_db: List[Tuple[List[List[int]], int]], 
                         item: int) -> List[Tuple[List[List[int]], int]]:
        """
        Create a projected database for a given item.
        
        Args:
            projected_db: Current projected database
            item: Item to project
            
        Returns:
            New projected database with sequences postfixed with the item
        """
        new_projected_db: List[Tuple[List[List[int]], int]] = []
        
        for sequence, pos in projected_db:
            # Find the first occurrence of item after position pos
            for i in range(pos, len(sequence)):
                # Check if item is in the current itemset
                if item in sequence[i]:
                    # Add projected sequence with position incremented
                    new_projected_db.append((sequence, i + 1))
                    break
        
        return new_projected_db
    
    def _mine_sequential_patterns(self, prefix: List[List[int]], 
                                 projected_db: List[Tuple[List[List[int]], int]], 
                                 pattern_length: int) -> None:
        """
        Mine sequential patterns recursively.
        
        Args:
            prefix: Current prefix pattern
            projected_db: Current projected database
            pattern_length: Length of the current pattern
        """
        # Find all frequent items in the projected database
        frequent_items = self._find_frequent_items(projected_db)
        
        # If no frequent items, return
        if not frequent_items:
            return
            
        # Initialize patterns for this length if not already
        if pattern_length not in self.patterns:
            self.patterns[pattern_length] = []
        
        # For each frequent item, extend the pattern
        for item, support in frequent_items.items():
            # Create new pattern by adding item to the prefix
            new_prefix = prefix.copy()
            
            # Append item as a new itemset
            new_prefix.append([item])
            
            # Add pattern to results
            self.patterns[pattern_length].append((new_prefix, support))
            
            # Create projected database for the new pattern
            new_projected_db = self._project_database(projected_db, item)
            
            # Recursively mine with the new prefix
            self._mine_sequential_patterns(new_prefix, new_projected_db, pattern_length + 1)
    
    def mine(self, sequences: List[List[List[int]]]) -> Dict[int, List[Tuple[List[List[int]], int]]]:
        """
        Mine sequential patterns from the input sequences.
        
        Args:
            sequences: Input sequences
            
        Returns:
            Dictionary of patterns grouped by length
        """
        start_time = time.time()
        self.total_sequences = len(sequences)
        self.min_support = int(self.min_support_ratio * self.total_sequences)
        
        if self.min_support < 1:
            self.min_support = 1
            
        print(f"Mining with minimum support count: {self.min_support} ({self.min_support_ratio:.2%})")
        
        # Initialize empty patterns dictionary
        self.patterns = {}
        
        # Create initial projected database
        initial_projected_db = [(seq, 0) for seq in sequences]
        
        # Start mining with an empty prefix
        self._mine_sequential_patterns([], initial_projected_db, 1)
        
        end_time = time.time()
        self._print_results(end_time - start_time)
        
        return self.patterns
    
    def _print_results(self, execution_time: float) -> None:
        """
        Print mining results.
        
        Args:
            execution_time: Time taken to mine patterns
        """
        total_patterns = sum(len(patterns) for patterns in self.patterns.values())
        
        print("\nPrefixSpan Results:")
        print(f"Total frequent sequential patterns found: {total_patterns}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Print pattern count by length
        print("\nPatterns by length:")
        for length, patterns in sorted(self.patterns.items()):
            print(f"  Length {length}: {len(patterns)} patterns")
        
        # Print detailed patterns if verbose
        if self.verbose and total_patterns > 0:
            print("\nFrequent sequential patterns:")
            for length, patterns in sorted(self.patterns.items()):
                print(f"\nLength {length} patterns:")
                for i, (pattern, support) in enumerate(patterns):
                    # Format pattern for readability
                    pattern_str = " -> ".join(str(itemset) for itemset in pattern)
                    print(f"  Pattern {i+1}: {pattern_str} (support: {support}, {support/self.total_sequences:.2%})")


def main() -> None:
    """Main function to run PrefixSpan algorithm from command line."""
    parser = argparse.ArgumentParser(description='Run PrefixSpan algorithm for sequential pattern mining.')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file')
    parser.add_argument('--min_support', type=float, default=0.1,
                        help='Minimum support ratio (0 to 1, default: 0.1)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print detailed information about extracted patterns')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.dataset}...")
    try:
        sequences = load_data(args.dataset)
        print(f"Loaded {len(sequences)} sequences successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create and run PrefixSpan
    prefixspan = PrefixSpan(min_support_ratio=args.min_support, verbose=args.verbose)
    prefixspan.mine(sequences)


if __name__ == "__main__":
    main()