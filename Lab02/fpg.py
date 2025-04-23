import math
import time
from collections import defaultdict, namedtuple

class FPTrNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None
        
    def increment(self, count):
        self.count += count
        
class FPGrowth:
    def __init__(self, transactions, min_support):
        self.transactions = transactions
        self.min_support = min_support
        self.min_support_count = math.ceil(len(transactions) * min_support)
        self.header_tab = {}
        self.freq_itemsets = []
        
    def build_header_table(self):
        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        self.header_tab = {item: [count, None] for item, count in item_counts.items() if count >= self.min_support_count}
        
    def build_fp_tree(self):
        self.build_header_table()
        root = FPTrNode(None, 1, None)
        
        for transaction in self.transactions:
            return
        
        