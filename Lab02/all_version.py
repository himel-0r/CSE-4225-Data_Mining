import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from itertools import combinations


class FrequentItemsetMining:
    def __init__(self, filename, min_support_percentage=25):
        self.filename = filename
        self.min_support_percentage = min_support_percentage
        self.transactions = []
        self.item_counts = Counter()
        self.min_support = 0
        self.load_data()
        
    def load_data(self):
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    items = line.strip().split()
                    self.transactions.append(set(items))
                    for item in items:
                        self.item_counts[item] += 1
            self.min_support = math.ceil(len(self.transactions) * self.min_support_percentage / 100)
            print(f"Loaded {len(self.transactions)} transactions")
            print(f"Minimum support count: {self.min_support} ({self.min_support_percentage}% of transactions)")
        except FileNotFoundError:
            print(f"Error: File {self.filename} not found.")
            exit(1)

    def apriori(self):
        all_frequent_itemsets = {}
        level_times = {}
        start_time = time.time()
        L1 = {frozenset([item]): count for item, count in self.item_counts.items() 
              if count >= self.min_support}
        level_times[1] = time.time() - start_time
        all_frequent_itemsets[1] = L1
        print(f"L1: Found {len(L1)} frequent 1-itemsets in {level_times[1]:.4f} seconds")
        k = 2
        Lk_minus_1 = L1
        while Lk_minus_1:
            start_time = time.time()
            Ck = self.apriori_gen(Lk_minus_1, k)
            itemset_counts = defaultdict(int)
            for transaction in self.transactions:
                for candidate in Ck:
                    if candidate.issubset(transaction):
                        itemset_counts[candidate] += 1
            Lk = {itemset: count for itemset, count in itemset_counts.items() 
                  if count >= self.min_support}
            level_times[k] = time.time() - start_time
            if Lk:
                all_frequent_itemsets[k] = Lk
                print(f"L{k}: Found {len(Lk)} frequent {k}-itemsets in {level_times[k]:.4f} seconds")
                Lk_minus_1 = Lk
                k += 1
            else:
                break
        return all_frequent_itemsets, level_times
    
    def apriori_gen(self, Lk_minus_1, k):
        candidates = set()
        prev_frequent_itemsets = list(Lk_minus_1.keys())
        for i in range(len(prev_frequent_itemsets)):
            for j in range(i+1, len(prev_frequent_itemsets)):
                itemset1 = list(prev_frequent_itemsets[i])
                itemset2 = list(prev_frequent_itemsets[j])
                itemset1.sort()
                itemset2.sort()
                if itemset1[:k-2] == itemset2[:k-2]:
                    candidate = frozenset(prev_frequent_itemsets[i] | prev_frequent_itemsets[j])
                    all_subsets_frequent = True
                    for subset in combinations(candidate, k-1):
                        if frozenset(subset) not in Lk_minus_1:
                            all_subsets_frequent = False
                            break
                    if all_subsets_frequent:
                        candidates.add(candidate)
        return candidates

    class FPNode:
        def __init__(self, item, count=1, parent=None):
            self.item = item
            self.count = count
            self.parent = parent
            self.children = {}
            self.link = None
            
    def fp_growth(self):
        start_time = time.time()
        header_table = {item: [count, None] for item, count in self.item_counts.items() 
                        if count >= self.min_support}
        sorted_items = sorted(header_table.keys(), key=lambda x: header_table[x][0], reverse=True)
        root = self.FPNode(None)
        for transaction in self.transactions:
            filtered_items = [item for item in sorted_items if item in transaction]
            if filtered_items:
                self._insert_tree(filtered_items, root, header_table)
        tree_build_time = time.time() - start_time
        print(f"FP-tree built in {tree_build_time:.4f} seconds")
        mine_start_time = time.time()
        frequent_patterns = {}
        self._mine_fp_tree(root, header_table, set(), frequent_patterns)
        mining_time = time.time() - mine_start_time
        total_time = time.time() - start_time
        print(f"FP-Growth: Patterns mined in {mining_time:.4f} seconds")
        print(f"FP-Growth: Total execution time: {total_time:.4f} seconds")
        level_patterns = {}
        for pattern, count in frequent_patterns.items():
            level = len(pattern)
            if level not in level_patterns:
                level_patterns[level] = {}
            level_patterns[level][pattern] = count
        return level_patterns, total_time
        
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
    
    def _mine_fp_tree(self, tree, header_table, base_pattern, frequent_patterns):
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
                            if cond_header_table[path_item][1] is None:
                                cond_header_table[path_item][1] = new_node
                            else:
                                link_node = cond_header_table[path_item][1]
                                while link_node.link is not None:
                                    link_node = link_node.link
                                link_node.link = new_node
                        current = current.children[path_item]
            self._mine_fp_tree(cond_tree, cond_header_table, new_pattern, frequent_patterns)


def visualize_comparison(apriori_results, fp_growth_results):
    all_levels = sorted(set(list(apriori_results.keys()) + list(fp_growth_results.keys())))
    apriori_counts = [len(apriori_results.get(level, {})) for level in all_levels]
    fp_growth_counts = [len(fp_growth_results.get(level, {})) for level in all_levels]
    bar_width = 0.35
    index = range(len(all_levels))
    plt.figure(figsize=(10, 6))
    bar1 = plt.bar([i - bar_width/2 for i in index], apriori_counts, bar_width, 
                   label='Apriori', color='blue', alpha=0.7)
    bar2 = plt.bar([i + bar_width/2 for i in index], fp_growth_counts, bar_width, 
                   label='FP-Growth', color='green', alpha=0.7)
    plt.xlabel('Itemset Size')
    plt.ylabel('Number of Frequent Itemsets')
    plt.title('Comparison of Frequent Itemsets Found by Algorithm')
    plt.xticks(index, [f'L{level}' for level in all_levels])
    plt.legend()
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom')
    add_labels(bar1)
    add_labels(bar2)
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.show()


def main():
    mining = FrequentItemsetMining('mushroom.dat')
    
    print("\n=== APRIORI ALGORITHM ===")
    apriori_results, apriori_times = mining.apriori()
    
    print("\n=== FP-GROWTH ALGORITHM ===")
    fp_growth_results, fp_growth_time = mining.fp_growth()
    
    print("\n=== SUMMARY ===")
    print("Apriori Algorithm:")
    total_apriori_time = sum(apriori_times.values())
    print(f"Total execution time: {total_apriori_time:.4f} seconds")
    
    print("\nFP-Growth Algorithm:")
    print(f"Total execution time: {fp_growth_time:.4f} seconds")
    
    improvement = (total_apriori_time - fp_growth_time) / total_apriori_time * 100
    print(f"\nFP-Growth is {improvement:.2f}% faster than Apriori")
    
    print("\nGenerating comparison chart...")
    visualize_comparison(apriori_results, fp_growth_results)
    print("Chart saved as 'algorithm_comparison.png'")


if __name__ == "__main__":
    main()
