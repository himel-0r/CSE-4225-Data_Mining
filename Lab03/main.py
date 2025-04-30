import time
import math
import matplotlib.pyplot as plt
import psutil
import os
from collections import defaultdict, Counter
from itertools import combinations

class FrequentItemsetMining:
    def __init__(self, filename, min_support_percentage):
        self.filename = filename
        self.min_support_percentage = min_support_percentage
        self.transactions = []
        self.item_counts = Counter()
        self.min_support = 0
        self.load_data()
        
    def load_data(self):
        with open(self.filename, 'r') as f:
                for line in f:
                    items = line.strip().split()
                    self.transactions.append(set(items))
                    for item in items:
                        self.item_counts[item] += 1
            
        # minimum support count threshold based on percentage
        self.min_support = math.ceil(len(self.transactions) * self.min_support_percentage / 100)
        print(f"Loaded {len(self.transactions)} transactions")
        print(f"Minimum support count: {self.min_support} ({self.min_support_percentage}% of transactions)")
        
    def apriori(self):
        all_frequent_itemsets = {}
        level_times = {}
        
        # Tracking memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Level 1
        start_time = time.time()
        L1 = {frozenset([item]): count for item, count in self.item_counts.items() 
              if count >= self.min_support}
        level_times[1] = time.time() - start_time
        all_frequent_itemsets[1] = L1
        
        print(f"L1: Found {len(L1)} frequent 1-itemsets in {level_times[1]:.4f} seconds")
        
        k = 2
        Lk_minus_1 = L1
        
        # All level finding
        while Lk_minus_1:
            start_time = time.time()
            
            # Generate candidate k-itemsets from frequent (k-1)-itemsets
            Ck = self.apriori_gen(Lk_minus_1, k)
            
            # Count support for each candidate
            itemset_counts = defaultdict(int)
            for transaction in self.transactions:
                for candidate in Ck:
                    if candidate.issubset(transaction):
                        itemset_counts[candidate] += 1
            
            # Filter candidates by minimum support threshold
            Lk = {itemset: count for itemset, count in itemset_counts.items() 
                  if count >= self.min_support}
            
            level_times[k] = time.time() - start_time
            
            # If we found frequent k-itemsets, store them and continue
            if Lk:
                all_frequent_itemsets[k] = Lk
                print(f"L{k}: Found {len(Lk)} frequent {k}-itemsets in {level_times[k]:.4f} seconds")
                Lk_minus_1 = Lk
                k += 1
            else:
                break
            
        # Calculate peak memory usage
        peak_memory = process.memory_info().rss / (1024 * 1024) - initial_memory
        if peak_memory < 0:
            peak_memory = 0
        print(f"Apriori peak memory usage: {peak_memory:.4f} MB")
        
        return all_frequent_itemsets, level_times, peak_memory
    
    def apriori_gen(self, Lk_minus_1, k):
        candidates = set()
        
        # Convert dictionary keys to list for easier indexing
        prev_frequent_itemsets = list(Lk_minus_1.keys())
        
        # Generate candidates using the join step
        for i in range(len(prev_frequent_itemsets)):
            for j in range(i+1, len(prev_frequent_itemsets)):
                itemset1 = list(prev_frequent_itemsets[i])
                itemset2 = list(prev_frequent_itemsets[j])
                
                # Sort to ensure consistent comparison
                itemset1.sort()
                itemset2.sort()
                
                # If first k-2 items are the same, join them
                if itemset1[:k-2] == itemset2[:k-2]:
                    # Create a new candidate by union
                    candidate = frozenset(prev_frequent_itemsets[i] | prev_frequent_itemsets[j])
                    
                    # Prune step: check if all (k-1)-subsets are frequent
                    all_subsets_frequent = True
                    for subset in combinations(candidate, k-1):
                        if frozenset(subset) not in Lk_minus_1:
                            all_subsets_frequent = False
                            break
                    
                    if all_subsets_frequent:
                        candidates.add(candidate)
        
        return candidates
    
    class FPNode:
        """Node in the FP-tree"""
        def __init__(self, item, count=1, parent=None):
            self.item = item
            self.count = count
            self.parent = parent
            self.children = {}
            self.link = None
            
    def fp_growth(self):
        # Track memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        
        header_table = {item: [count, None] for item, count in self.item_counts.items() 
                        if count >= self.min_support}
        
        sorted_items = sorted(header_table.keys(), key=lambda x: header_table[x][0], reverse=True)
        
        # Build the FP-tree
        root = self.FPNode(None)
        
        for transaction in self.transactions:
            # Filter and sort transaction items based on frequency
            filtered_items = [item for item in sorted_items if item in transaction]
            
            if filtered_items:
                self._insert_tree(filtered_items, root, header_table)
        
        tree_build_time = time.time() - start_time
        print(f"FP-tree built in {tree_build_time:.4f} seconds")
        
        # Frequent patterns mining
        mine_start_time = time.time()
        frequent_patterns = {}
        self._mine_fp_tree(root, header_table, set(), frequent_patterns)
        mining_time = time.time() - mine_start_time
        
        total_time = time.time() - start_time
        print(f"FP-Growth: Patterns mined in {mining_time:.4f} seconds")
        print(f"FP-Growth: Total execution time: {total_time:.4f} seconds")
        
        peak_memory = process.memory_info().rss / (1024 * 1024) - initial_memory
        if peak_memory < 0:
            peak_memory = 0
        print(f"FP-Growth peak memory usage: {peak_memory:.2f} MB")
        
        level_patterns = {}
        for pattern, count in frequent_patterns.items():
            level = len(pattern)
            if level not in level_patterns:
                level_patterns[level] = {}
            level_patterns[level][pattern] = count
            
        return level_patterns, total_time, peak_memory
    
    def _insert_tree(self, items, node, header_table):
        if not items:
            return
            
        item = items[0]
        
        if item in node.children:
            node.children[item].count += 1
        else:
            new_node = self.FPNode(item, 1, node)
            node.children[item] = new_node
            
            if header_table[item][1] is None:
                header_table[item][1] = new_node
            else:
                current = header_table[item][1]
                while current.link is not None:
                    current = current.link
                current.link = new_node
                
        self._insert_tree(items[1:], node.children[item], header_table)
    
    def _mine_fp_tree(self, header_table, base_pattern, frequent_patterns):
        for item in sorted(header_table.keys(), key=lambda x: header_table[x][0]):
            new_pattern = base_pattern.copy()
            new_pattern.add(item)
            
            pattern_support = header_table[item][0]
            frequent_patterns[frozenset(new_pattern)] = pattern_support
            
            conditional_pattern_base = []
            
            node = header_table[item][1]
            
            while node is not None:
                path = []
                support = node.count
                
                parent = node.parent
                while parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                
                if path:
                    conditional_pattern_base.append((path, support))
                
                node = node.link
            
            if not conditional_pattern_base:
                continue
                
            cond_header_table = {}
            
            for path, count in conditional_pattern_base:
                for path_item in path:
                    if path_item not in cond_header_table:
                        cond_header_table[path_item] = [0, None]
                    cond_header_table[path_item][0] += count
            
            cond_header_table = {k: v for k, v in cond_header_table.items() 
                                if v[0] >= self.min_support}
            
            if not cond_header_table:
                continue
                
            cond_tree = self.FPNode(None)
            
            for path, count in conditional_pattern_base:
                filtered_path = [p for p in path if p in cond_header_table]
                filtered_path.sort(key=lambda x: cond_header_table[x][0], reverse=True)
                
                if filtered_path:
                    current = cond_tree
                    for path_item in filtered_path:
                        if path_item in current.children:
                            current.children[path_item].count += count
                        else:
                            new_node = self.FPNode(path_item, count, current)
                            current.children[path_item] = new_node
                            
                            # Update header table
                            if cond_header_table[path_item][1] is None:
                                cond_header_table[path_item][1] = new_node
                            else:
                                link_node = cond_header_table[path_item][1]
                                while link_node.link is not None:
                                    link_node = link_node.link
                                link_node.link = new_node
                                
                        current = current.children[path_item]
            
            # Recursively mine the conditional FP-tree
            self._mine_fp_tree(cond_tree, cond_header_table, new_pattern, frequent_patterns)

def main(filename, min_sup=25, max_sup=60, gaps=5):
    support_values = list(range(min_sup, max_sup + 1, gaps))
    apriori_times = []
    fp_growth_times = []
    apriori_memories = []
    fp_growth_memories = []

    for support in support_values:
        print(f"\nRunning for min_support = {support}%")
        mining = FrequentItemsetMining(filename, min_support_percentage=support)

        print("  Running Apriori...")
        _, apriori_time_dict, apriori_memory = mining.apriori()
        apriori_total_time = sum(apriori_time_dict.values())
        apriori_times.append(apriori_total_time)
        apriori_memories.append(apriori_memory)

        print("  Running FP-Growth...")
        _, fp_growth_time, fp_growth_memory = mining.fp_growth()
        fp_growth_times.append(fp_growth_time)
        fp_growth_memories.append(fp_growth_memory)

    # Execution Time Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(support_values, apriori_times, 'bo-', linewidth=2, markersize=8, label='Apriori Time')
    plt.plot(support_values, fp_growth_times, 'ro-', linewidth=2, markersize=8, label='FP-Growth Time')
    plt.xlabel('Minimum Support (%)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Minimum Support (connect)') ###
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('connect_time.png') ###
    plt.show()

    # Memory Usage Comparison 
    plt.figure(figsize=(12, 6))
    plt.plot(support_values, apriori_memories, 'go-', linewidth=2, markersize=8, label='Apriori Memory')
    plt.plot(support_values, fp_growth_memories, 'mo-', linewidth=2, markersize=8, label='FP-Growth Memory')
    plt.xlabel('Minimum Support (%)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Minimum Support (connect)') ###
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('connect_memory.png') ###
    plt.show()

if __name__ == "__main__":
    st_time = time.time()
    main('connect.dat', 95, 100, 1)
    ed_time = time.time()
    
    print("Total taken time = ", ed_time - st_time)