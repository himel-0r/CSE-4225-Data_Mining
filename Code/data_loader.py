from typing import List, Dict, Tuple, Set, Union
from pathlib import Path


def load_data(file_path):
    sequences = []
    current_sequence = []
    current_event = set()

    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            for token in tokens:
                number = int(token)
                
                if number == -2:
                    if current_sequence:
                        sequences.append(current_sequence)
                        current_sequence = []
                elif number == -1:
                    if current_event:
                        current_sequence.append(current_event)
                        current_event = set()
                else:
                    current_event.add(number)
    return sequences

def get_statistics(sequences: List[List[List[int]]]) -> Dict:
    total_sequences = len(sequences)
    total_itemsets = sum(len(seq) for seq in sequences)
    total_items = sum(sum(len(itemset) for itemset in seq) for seq in sequences)
    
    unique_items = set()
    for seq in sequences:
        for itemset in seq:
            for item in itemset:
                unique_items.add(item)
    
    avg_seq_length = total_itemsets / total_sequences if total_sequences > 0 else 0
    avg_itemset_size = total_items / total_itemsets if total_itemsets > 0 else 0
    
    return {
        "total_sequences": total_sequences,
        "total_itemsets": total_itemsets,
        "total_items": total_items,
        "unique_items": len(unique_items),
        "avg_sequence_length": avg_seq_length,
        "avg_itemset_size": avg_itemset_size
    }


if __name__ == "__main__":
    print(load_data('data/eshop.dat'))