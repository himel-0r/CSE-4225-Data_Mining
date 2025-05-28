from typing import List, Dict, Tuple, Set, Union
from pathlib import Path


def load_data(file_path: str) -> List[List[List[int]]]:
    """
    Load sequential pattern data from a file.
    
    Args:
        file_path (str): Path to the data file
    
    Returns:
        List[List[List[int]]]: A list of sequences, where each sequence is a list of itemsets,
                               and each itemset is a list of integers.
                               
    Format:
        - Each time series event is separated by -1
        - Each sequence of events is separated by -2
        - For example, the following data:
          10 192 108 275 -1 10 192 315 -1 -2
          80 169 188 -1 -2
          Will be parsed as:
          [
              [[10, 192, 108, 275], [10, 192, 315]],
              [[80, 169, 188]]
          ]
    """
    sequences = []
    current_sequence = []
    current_itemset: List = []
    
    # Handle file with comments
    path = Path(file_path)
    
    with open(path, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.strip().startswith('//'):
                continue
                
            # Process each number in the line
            for token in line.strip().split():
                if token == '-1':
                    # End of an itemset
                    if current_itemset:
                        current_sequence.append(current_itemset)
                        current_itemset = []
                elif token == '-2':
                    # End of a sequence
                    if current_sequence:
                        sequences.append(current_sequence)
                        current_sequence = []
                else:
                    # Add item to current itemset
                    try:
                        current_itemset.append(int(token))
                    except ValueError:
                        # Skip non-integer values
                        pass
    
    # Handle last sequence if file doesn't end with -2
    if current_itemset:
        current_sequence.append(current_itemset)
    if current_sequence:
        sequences.append(current_sequence)
        
    return sequences


def get_statistics(sequences: List[List[List[int]]]) -> Dict:
    """
    Get statistics about the loaded dataset.
    
    Args:
        sequences (List[List[List[int]]]): Loaded sequences
    
    Returns:
        Dict: Dictionary containing statistics about the dataset
    """
    total_sequences = len(sequences)
    total_itemsets = sum(len(seq) for seq in sequences)
    total_items = sum(sum(len(itemset) for itemset in seq) for seq in sequences)
    
    # Get unique items
    unique_items = set()
    for seq in sequences:
        for itemset in seq:
            for item in itemset:
                unique_items.add(item)
    
    # Calculate average sequence length
    avg_seq_length = total_itemsets / total_sequences if total_sequences > 0 else 0
    
    # Calculate average itemset size
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
    # Example usage
    print(load_data('data/eshop.dat'))
    
    
    # # Relative path from the script location
    # data_folder = os.path.join(os.path.dirname(__file__), "data")
    # eshop_data_path = os.path.join(data_folder, "eshop.dat")
    
    # # Load the data
    # sequences = load_data(eshop_data_path)
    
    # # Print some statistics
    # stats = get_statistics(sequences)
    # print(f"Dataset Statistics:")
    # print(f"Total sequences: {stats['total_sequences']}")
    # print(f"Total itemsets: {stats['total_itemsets']}")
    # print(f"Total items: {stats['total_items']}")
    # print(f"Unique items: {stats['unique_items']}")
    # print(f"Average sequence length: {stats['avg_sequence_length']:.2f} itemsets")
    # print(f"Average itemset size: {stats['avg_itemset_size']:.2f} items")
    
    # # Print first few sequences for visualization
    # print("\nFirst 2 sequences:")
    # for i, seq in enumerate(sequences[:2]):
    #     print(f"Sequence {i+1}:")
    #     for j, itemset in enumerate(seq):
    #         print(f"  Itemset {j+1}: {itemset}")
            
            
