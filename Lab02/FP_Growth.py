import time
import math
from collections import defaultdict, namedtuple

class FPTreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None

    def increment(self, count):
        self.count += count


class FPGrowth:
    def __init__(self, transactions, min_support_ratio):
        self.transactions = transactions
        self.min_support_ratio = min_support_ratio
        self.min_support_count = math.ceil(len(transactions) * min_support_ratio)
        self.header_table = {}
        self.freq_itemsets = []

    def build_header_table(self):
        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        self.header_table = {item: [count, None] for item, count in item_counts.items() if count >= self.min_support_count}

    def build_fp_tree(self):
        self.build_header_table()
        root = FPTreeNode(None, 1, None)

        for transaction in self.transactions:
            # Keep only frequent items and sort by frequency descending
            filtered = [item for item in transaction if item in self.header_table]
            sorted_items = sorted(filtered, key=lambda item: (-self.header_table[item][0], item))
            self.insert_tree(sorted_items, root)

        return root

    def insert_tree(self, items, node):
        if not items:
            return
        first_item = items[0]
        if first_item in node.children:
            node.children[first_item].increment(1)
        else:
            new_node = FPTreeNode(first_item, 1, node)
            node.children[first_item] = new_node

            # Update header table
            if self.header_table[first_item][1] is None:
                self.header_table[first_item][1] = new_node
            else:
                current = self.header_table[first_item][1]
                while current.link is not None:
                    current = current.link
                current.link = new_node

        self.insert_tree(items[1:], node.children[first_item])

    def ascend_tree(self, node):
        path = []
        while node.parent and node.parent.item is not None:
            node = node.parent
            path.append(node.item)
        return path

    def find_prefix_paths(self, base_pat):
        conditional_patterns = []
        node = self.header_table[base_pat][1]
        while node is not None:
            path = self.ascend_tree(node)
            if path:
                conditional_patterns.append((path, node.count))
            node = node.link
        return conditional_patterns

    def mine_tree(self, tree, header_table, prefix):
        sorted_items = sorted(header_table.items(), key=lambda item: (item[1][0], item[0]))
        for base_pat, (support, node) in sorted_items:
            new_freq_set = prefix.copy()
            new_freq_set.add(base_pat)
            self.freq_itemsets.append((new_freq_set, support))

            conditional_patterns = self.find_prefix_paths(base_pat)
            conditional_transactions = []
            for path, count in conditional_patterns:
                for _ in range(count):
                    conditional_transactions.append(set(path))

            if conditional_transactions:
                conditional_miner = FPGrowth(conditional_transactions, self.min_support_ratio)
                conditional_tree = conditional_miner.build_fp_tree()
                if conditional_tree:
                    conditional_miner.mine_tree(conditional_tree, conditional_miner.header_table, new_freq_set)
                    self.freq_itemsets.extend(conditional_miner.freq_itemsets)

    def run(self):
        root = self.build_fp_tree()
        if root:
            self.mine_tree(root, self.header_table, set())
        return self.freq_itemsets


def load_data_from_file(filename):
    transactions = []
    with open(filename, 'r') as file:
        for line in file:
            transaction = set(map(int, line.strip().split()))
            transactions.append(transaction)
    return transactions


def main(transactions, ratio):
    min_support_ratio = ratio * 0.01

    start_time = time.time()
    fp_growth = FPGrowth(transactions, min_support_ratio)
    freq_itemsets = fp_growth.run()

    print(f"Minimum support count: {fp_growth.min_support_count}\n")
    levels = defaultdict(int)
    for itemset, _ in freq_itemsets:
        levels[len(itemset)] += 1

    total = 0
    for k in sorted(levels):
        print(f"L_{k}: {levels[k]} itemsets")
        total += levels[k]
    print("Total = ", total)
    end_time = time.time()
    print("Total Taken Time = ", end_time - start_time)


if __name__ == "__main__":
    filename = 'mushroom.dat'
    transactions = load_data_from_file(filename)

    for ratio in range(50, 59, 5):
        print("Ratio: ", ratio, ": ")
        main(transactions, ratio)
        print("\n\n\n")
