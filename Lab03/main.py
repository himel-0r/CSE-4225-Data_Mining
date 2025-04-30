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
        
        # Step 1: Build header table (maps items to their frequency and last node in the linked list)
        header_table = {item: [count, None] for item, count in self.item_counts.items() 
                        if count >= self.min_support}
        
        # Sort items by frequency (descending)
        sorted_items = sorted(header_table.keys(), key=lambda x: header_table[x][0], reverse=True)
        
        # Step 2: Build the FP-tree
        root = self.FPNode(None)
        
        for transaction in self.transactions:
            # Filter and sort transaction items based on frequency
            filtered_items = [item for item in sorted_items if item in transaction]
            
            if filtered_items:
                self._insert_tree(filtered_items, root, header_table)
        
        tree_build_time = time.time() - start_time
        print(f"FP-tree built in {tree_build_time:.4f} seconds")
        
        # Step 3: Mine frequent patterns
        mine_start_time = time.time()
        frequent_patterns = {}
        self._mine_fp_tree(root, header_table, set(), frequent_patterns)
        mining_time = time.time() - mine_start_time
        
        total_time = time.time() - start_time
        print(f"FP-Growth: Patterns mined in {mining_time:.4f} seconds")
        print(f"FP-Growth: Total execution time: {total_time:.4f} seconds")
        
        # Calculate peak memory usage
        peak_memory = process.memory_info().rss / (1024 * 1024) - initial_memory
        if peak_memory < 0:
            peak_memory = 0
        print(f"FP-Growth peak memory usage: {peak_memory:.2f} MB")
        
        # Group patterns by size
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
        
        # If this item is already a child of the current node, increment its count
        if item in node.children:
            node.children[item].count += 1
        else:
            # Create a new node
            new_node = self.FPNode(item, 1, node)
            node.children[item] = new_node
            
            # Link it to the header table
            if header_table[item][1] is None:
                header_table[item][1] = new_node
            else:
                current = header_table[item][1]
                while current.link is not None:
                    current = current.link
                current.link = new_node
                
        # Recursively insert the rest of the items
        self._insert_tree(items[1:], node.children[item], header_table)
    
    def _mine_fp_tree(self, tree, header_table, base_pattern, frequent_patterns):
        # Process each item in the header table in reverse order (least frequent first)
        for item in sorted(header_table.keys(), key=lambda x: header_table[x][0]):
            # Current pattern is base_pattern + item
            new_pattern = base_pattern.copy()
            new_pattern.add(item)
            
            # Add this pattern to the result
            pattern_support = header_table[item][0]
            frequent_patterns[frozenset(new_pattern)] = pattern_support
            
            # Find conditional pattern base
            conditional_pattern_base = []
            
            # Follow the linked list for this item
            node = header_table[item][1]
            
            while node is not None:
                # Find path from this node to the root
                path = []
                support = node.count
                
                # Traverse up the tree to collect the path
                parent = node.parent
                while parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                
                if path:
                    conditional_pattern_base.append((path, support))
                
                # Move to the next node in the linked list
                node = node.link
            
            # Skip if no conditional pattern base
            if not conditional_pattern_base:
                continue
                
            # Build conditional FP-tree
            cond_header_table = {}
            
            # Count items in the conditional pattern base
            for path, count in conditional_pattern_base:
                for path_item in path:
                    if path_item not in cond_header_table:
                        cond_header_table[path_item] = [0, None]
                    cond_header_table[path_item][0] += count
            
            # Filter items by minimum support
            cond_header_table = {k: v for k, v in cond_header_table.items() 
                                if v[0] >= self.min_support}
            
            if not cond_header_table:
                continue
                
            # Build conditional FP-tree
            cond_tree = self.FPNode(None)
            
            for path, count in conditional_pattern_base:
                # Filter and sort items in path
                filtered_path = [p for p in path if p in cond_header_table]
                filtered_path.sort(key=lambda x: cond_header_table[x][0], reverse=True)
                
                if filtered_path:
                    # Insert the path into the conditional tree
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


# def visualize_comparison(apriori_results, fp_growth_results):
#     # Get all levels
#     all_levels = sorted(set(list(apriori_results.keys()) + list(fp_growth_results.keys())))
    
#     # Count itemsets for each level
#     apriori_counts = [len(apriori_results.get(level, {})) for level in all_levels]
#     fp_growth_counts = [len(fp_growth_results.get(level, {})) for level in all_levels]
    
#     # Create the bar chart
#     plt.figure(figsize=(10, 6))
    
#     # Set up bar positions
#     bar_width = 0.35
#     index = range(len(all_levels))
    
#     bar1 = plt.bar([i - bar_width/2 for i in index], apriori_counts, bar_width, 
#                    label='Apriori', color='blue', alpha=0.7)
#     bar2 = plt.bar([i + bar_width/2 for i in index], fp_growth_counts, bar_width, 
#                    label='FP-Growth', color='green', alpha=0.7)
    
#     # Add labels and title
#     plt.xlabel('Itemset Size')
#     plt.ylabel('Number of Frequent Itemsets')
#     plt.title('Comparison of Frequent Itemsets Found by Algorithm')
#     plt.xticks(index, [f'L{level}' for level in all_levels])
#     plt.legend()
    
#     # Add count labels on top of bars
#     def add_labels(bars):
#         for bar in bars:
#             height = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
#                      f'{int(height)}', ha='center', va='bottom')
    
#     add_labels(bar1)
#     add_labels(bar2)
    
#     # Save the bar chart
#     plt.tight_layout()
#     plt.savefig('algorithm_comparison.png')
    
#     # Create the line chart for the same data
#     plt.figure(figsize=(10, 6))
    
#     plt.plot(all_levels, apriori_counts, 'bo-', linewidth=2, markersize=8, label='Apriori')
#     plt.plot(all_levels, fp_growth_counts, 'go-', linewidth=2, markersize=8, label='FP-Growth')
    
#     # Add data labels
#     for i, v in enumerate(apriori_counts):
#         plt.text(all_levels[i], v + 0.5, str(v), ha='center')
#     for i, v in enumerate(fp_growth_counts):
#         plt.text(all_levels[i], v + 0.5, str(v), ha='center')
    
#     plt.xlabel('Itemset Size (Level)')
#     plt.ylabel('Number of Frequent Itemsets')
#     plt.title('Comparison of Frequent Itemsets Found (Line Chart)')
#     plt.xticks(all_levels, [f'L{level}' for level in all_levels])
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('itemset_comparison_line.png')
#     plt.show()

# def main(filename='mushroom.dat'):
#     """Main function to run the algorithms and compare results"""
#     # Initialize the mining object
#     mining = FrequentItemsetMining(filename)
    
#     print("\n=== APRIORI ALGORITHM ===")
#     apriori_results, apriori_times, apriori_memory = mining.apriori()
    
#     print("\n=== FP-GROWTH ALGORITHM ===")
#     fp_growth_results, fp_growth_time, fp_growth_memory = mining.fp_growth()
    
#     # Print summary
#     print("\n=== SUMMARY ===")
#     print("Apriori Algorithm:")
#     total_apriori_time = sum(apriori_times.values())
#     print(f"Total execution time: {total_apriori_time:.4f} seconds")
#     print(f"Memory usage: {apriori_memory:.2f} MB")
    
#     print("\nFP-Growth Algorithm:")
#     print(f"Total execution time: {fp_growth_time:.4f} seconds")
#     print(f"Memory usage: {fp_growth_memory:.2f} MB")
    
#     # Time improvement
#     time_improvement = (total_apriori_time - fp_growth_time) / total_apriori_time * 100
#     memory_improvement = (apriori_memory - fp_growth_memory) / apriori_memory * 100
#     print(f"\nFP-Growth is {time_improvement:.2f}% faster than Apriori")
#     print(f"FP-Growth uses {memory_improvement:.2f}% less memory than Apriori")
    
#     # Visualize comparison
#     print("\nGenerating comparison charts...")
#     visualize_comparison(apriori_results, fp_growth_results)
#     print("Chart saved as 'algorithm_comparison.png'")
    
#     # Generate performance comparison charts
#     visualize_performance(
#         {'Apriori': total_apriori_time, 'FP-Growth': fp_growth_time},
#         {'Apriori': apriori_memory, 'FP-Growth': fp_growth_memory}
#     )
#     print("Performance charts saved as 'time_comparison.png' and 'memory_comparison.png'")

# def visualize_performance(time_data, memory_data):
#     # Time comparison
#     plt.figure(figsize=(10, 6))
#     algorithms = list(time_data.keys())
#     times = list(time_data.values())
    
#     # Create line graph for time comparison
#     plt.plot(algorithms, times, 'ro-', linewidth=2, markersize=10)
    
#     # Add data labels
#     for i, v in enumerate(times):
#         plt.text(i, v + 0.05, f"{v:.4f}s", ha='center')
    
#     plt.ylabel('Execution Time (seconds)')
#     plt.title('Execution Time Comparison')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xticks(range(len(algorithms)), algorithms)
    
#     plt.tight_layout()
#     plt.savefig('time_comparison.png')
    
#     # Memory comparison
#     plt.figure(figsize=(10, 6))
#     memories = list(memory_data.values())
    
#     # Create line graph for memory comparison
#     plt.plot(algorithms, memories, 'bo-', linewidth=2, markersize=10)
    
#     # Add data labels
#     for i, v in enumerate(memories):
#         plt.text(i, v + 0.5, f"{v:.2f} MB", ha='center')
    
#     plt.ylabel('Memory Usage (MB)')
#     plt.title('Memory Usage Comparison')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xticks(range(len(algorithms)), algorithms)
    
#     plt.tight_layout()
#     plt.savefig('memory_comparison.png')
#     plt.show()

def benchmark_support_variation(filename, min_sup=25, max_sup=60, gaps=5):
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
    # main()
    st_time = time.time()
    benchmark_support_variation('connect.dat', 95, 100, 1)
    ed_time = time.time()
    
    print("Total taken time = ", ed_time - st_time)