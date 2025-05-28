from collections import defaultdict
import time
from typing import List, Tuple, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

class PrefixSpan:
    def __init__(self, min_support_ratio: float = 0.1, verbose: bool = False):
        self.min_support_ratio = min_support_ratio
        self.verbose = verbose
        self.patterns = {}
        self.total_sequences = 0
        self.min_support = 0

    def _find_frequent_items(self, projected_db: List[Tuple[List[List[str]], int]]) -> Dict[str, int]:
        item_count = defaultdict(int)

        for sequence, pos in projected_db:
            seen = set()
            for i in range(pos, len(sequence)):
                for item in sequence[i]:
                    if item not in seen:
                        item_count[item] += 1
                        seen.add(item)
                        break  # only one instance per sequence

        return {item: count for item, count in item_count.items() if count >= self.min_support}

    def _project_database(self, projected_db, item):
        new_projected_db = []
        for sequence, pos in projected_db:
            for i in range(pos, len(sequence)):
                if item in sequence[i]:
                    new_projected_db.append((sequence, i + 1))  # move to next itemset
                    break  # only first occurrence counts
        return new_projected_db

    def _project_database_itemset(self, projected_db, extended_itemset):
        new_projected_db = []
        support = 0

        for sequence, pos in projected_db:
            if pos < len(sequence):
                itemset = sequence[pos]
                if all(it in itemset for it in extended_itemset):
                    support += 1
                    new_projected_db.append((sequence, pos))  # stay in same itemset

        return support, new_projected_db

    def _mine_sequential_patterns(self, prefix: List[List[str]],
                                  projected_db: List[Tuple[List[List[str]], int]],
                                  pattern_length: int):
        frequent_items = self._find_frequent_items(projected_db)
        if not frequent_items:
            return

        if pattern_length not in self.patterns:
            self.patterns[pattern_length] = []

        for item, _ in frequent_items.items():
            # -------- Sequence Extension (new itemset)
            new_prefix = prefix + [[item]]
            new_projected_db = self._project_database(projected_db, item)
            support = len(new_projected_db)
            if support >= self.min_support:
                self.patterns[pattern_length].append((new_prefix, support))
                self._mine_sequential_patterns(new_prefix, new_projected_db, pattern_length + 1)

            # -------- Itemset Extension (add to last itemset)
            if prefix:
                last_itemset = prefix[-1]
                if item not in last_itemset and item > last_itemset[-1]:
                    extended_itemset = last_itemset + [item]
                    extended_prefix = prefix[:-1] + [extended_itemset]
                    support_itemset, new_proj_itemset_db = self._project_database_itemset(projected_db, extended_itemset)
                    if support_itemset >= self.min_support:
                        self.patterns[pattern_length].append((extended_prefix, support_itemset))
                        self._mine_sequential_patterns(extended_prefix, new_proj_itemset_db, pattern_length + 1)

    def mine(self, sequences: List[List[List[str]]]):
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

    def _print_results(self, execution_time: float):
        total_patterns = sum(len(pats) for pats in self.patterns.values())
        print("\nPrefixSpan Results:")
        print(f"Total frequent sequential patterns found: {total_patterns}")
        print(f"Execution time: {execution_time:.2f} seconds")

        print("\nPatterns by length:")
        for length, patterns in sorted(self.patterns.items()):
            print(f"  Length {length}: {len(patterns)} patterns")
            if self.verbose:
                for pattern, support in patterns:
                    pattern_str = " -> ".join(f"({','.join(p)})" for p in pattern)
                    print(f"    {pattern_str} (support: {support})")


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


if __name__ == "__main__":
    main('book.txt', 0.50)