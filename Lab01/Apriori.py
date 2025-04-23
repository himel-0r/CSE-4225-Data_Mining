import time
import math

class Apriori:
    def __init__(self, transactions, min_support_ratio, prev_time):
        self.transactions = transactions
        self.min_support_ratio = min_support_ratio
        self.min_support_count = math.ceil(len(transactions) * min_support_ratio)
        self.freq_itemsets = []
        self.levels = {}
        self.prev_time = prev_time

    def get_frequent_1_itemsets(self):
        item_count = {}
        for transaction in self.transactions:
            for item in transaction:
                item_count[item] = item_count.get(item, 0) + 1
        L1 = []
        for item, count in item_count.items():
            if count >= self.min_support_count:
                L1.append(frozenset([item]))
        return L1

    def get_itemset_support_count(self, itemset):
        count = 0
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count

    def generate_candidates(self, prev_freq_itemsets, k):
        candidates = set()
        n = len(prev_freq_itemsets)
        prev_freq_list = list(prev_freq_itemsets)
        for i in range(n):
            for j in range(i + 1, n):
                l1 = list(prev_freq_list[i])
                l2 = list(prev_freq_list[j])
                l1.sort()
                l2.sort()
                if l1[:k-2] == l2[:k-2]:  # join step
                    candidate = prev_freq_list[i] | prev_freq_list[j]
                    if len(candidate) == k:
                        candidates.add(candidate)
        return list(candidates)

    def prune_candidates(self, candidates, prev_freq_itemsets):
        pruned = []
        prev_freq_set = set(prev_freq_itemsets)
        for candidate in candidates:
            all_subsets = [candidate - frozenset([item]) for item in candidate]
            if all(subset in prev_freq_set for subset in all_subsets):
                pruned.append(candidate)
        return pruned

    def apriori_gen(self, max_level):
        L1 = self.get_frequent_1_itemsets()
        current_L = L1
        k = 1
        while current_L and k <= max_level:
            self.freq_itemsets.extend(current_L)
            self.levels[k] = current_L
            candidates = self.generate_candidates(current_L, k + 1)
            print("Candidate: ", len(candidates), end="   ")
            before = len(candidates)
            candidates = self.prune_candidates(candidates, current_L)
            print("Pruned: ", before - len(candidates))
            next_L = []
            for candidate in candidates:
                count = self.get_itemset_support_count(candidate)
                if count >= self.min_support_count:
                    next_L.append(candidate)
            current_L = next_L
            k += 1
            now_time = time.time()
            print("K = ", k, " time = ", now_time-self.prev_time)
            self.prev_time = now_time
            

    def run(self, max_level):
        self.apriori_gen(max_level)
        return self.freq_itemsets


def load_data_from_file(filename):
    transactions = []
    with open(filename, 'r') as file:
        for line in file:
            transaction = set(map(int, line.strip().split()))
            transactions.append(transaction)
    return transactions


def main(transactions, ratio):
    min_support_ratio = ratio
    max_level = 100

    start_time = time.time()
    prev_time = start_time

    apriori = Apriori(transactions, min_support_ratio * .01, prev_time)
    frequent_itemsets = apriori.run(max_level)

    print(f"Minimum support count: {apriori.min_support_count}\n")
    sum_val = 0

    for i in range(1, max_level + 1):
        count = len(apriori.levels.get(i, []))
        if count == 0:
            break
        print(f"L_{i}: {count} itemsets")
        sum_val += count

    print("Total = ", sum_val)

    end_time = time.time()

    print("Total Taken Time = ", end_time-start_time)
    


if __name__ == "__main__":
    filename = 'mushroom.dat'
    transactions = load_data_from_file(filename)
    
    for ratio in range (15, 96, 5):
        print("Ratio: ", ratio, ": ")
        main(transactions, ratio)
        print("\n\n\n")