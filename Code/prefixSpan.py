import sys
import os
import argparse
from typing import List, Dict, Tuple, Set, Any
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

class PrefixSpan:
    def __init__(self, min_support_ratio: float = 0.1, verbose: bool = False):
        self.min_support_ratio = min_support_ratio
        self.verbose = verbose
        self.patterns = {}
        self.total_sequences = 0
        self.min_support = 0
    
    def _find_frequent_items(self, projected_db):
        item_count = defaultdict(int)
        
        for sequence, pos in projected_db:
            remaining = sequence[pos:]
            if not remaining:
                continue
                
            seen_items: Set[int] = set()
            
            for itemset in remaining:
                for item in itemset:
                    if item not in seen_items:
                        item_count[item] += 1
                        seen_items.add(item)
        
        frequent_items = {item: count for item, count in item_count.items() 
                         if count >= self.min_support}
        
        return frequent_items
    
    def _project_database(self, projected_db, item):
        new_projected_db = []
        
        for sequence, pos in projected_db:
            for i in range(pos, len(sequence)):
                if item in sequence[i]:
                    new_projected_db.append((sequence, i + 1))
                    break
        
        return new_projected_db
    
    def _mine_sequential_patterns(self, prefix, 
                                 projected_db, 
                                 pattern_length) -> None:

        frequent_items = self._find_frequent_items(projected_db)
        
        if not frequent_items:
            return
        
        if pattern_length not in self.patterns:
            self.patterns[pattern_length] = []
        
        for item, support in frequent_items.items():
            new_prefix = prefix.copy()
            new_prefix.append([item])
            self.patterns[pattern_length].append((new_prefix, support))
            new_projected_db = self._project_database(projected_db, item)
            # print(prefix)
            # print(new_projected_db)
            self._mine_sequential_patterns(new_prefix, new_projected_db, pattern_length + 1)
            
            # if prefix:
                # print(prefix)
    
    def mine(self, sequences):
        start_time = time.time()
        self.total_sequences = len(sequences)
        self.min_support = int(self.min_support_ratio * self.total_sequences)
        
        if self.min_support < 1:
            self.min_support = 1
            
        print(f"Mining with minimum support count: {self.min_support} ({self.min_support_ratio:.2%})")
        
        self.patterns = {}
        initial_projected_db = [(seq, 0) for seq in sequences]
        
        self._mine_sequential_patterns([], initial_projected_db, 1)
        
        end_time = time.time()
        self._print_results(end_time - start_time)
        
        return self.patterns
    
    def _print_results(self, execution_time: float) -> None:
        total_patterns = sum(len(patterns) for patterns in self.patterns.values())
        
        print("\nPrefixSpan Results:")
        print(f"Total frequent sequential patterns found: {total_patterns}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Print pattern count by length
        print("\nPatterns by length:")
        for length, patterns in sorted(self.patterns.items()):
            print(f"  Length {length}: {len(patterns)} patterns")
            # print(patterns)
        
        # Print detailed patterns if verbose
        if self.verbose and total_patterns > 0:
            print("\nFrequent sequential patterns:")
            for length, patterns in sorted(self.patterns.items()):
                print(f"\nLength {length} patterns:")
                for i, (pattern, support) in enumerate(patterns):
                    pattern_str = " -> ".join(str(itemset) for itemset in pattern)
                    print(f"  Pattern {i+1}: {pattern_str} (support: {support}, {support/self.total_sequences:.2%})")


def main(filename, min_sup):
    print(f"Loading data from {filename}...")
    try:
        sequences = load_data(filename)
        print(f"Loaded {len(sequences)} sequences successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    prefixspan = PrefixSpan(min_support_ratio=min_sup, verbose=None)
    prefixspan.mine(sequences)


def BMS1():
    filename = r"BMS1_spmf.txt"
    sups = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    
    for sup in sups:
        main(filename, sup)
        
def BIKE():
    filename = r"BIKE.txt"
    sups = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    
    for sup in sups:
        main(filename, sup)
        
def SIGN():
    filename = r"SIGN.txt"
    sups = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    
    for sup in sups:
        main(filename, sup)
        
if __name__ == "__main__":
    # main('book.txt', 0.5)
    # BMS1()
    # BIKE()
    SIGN()