import copy
import numpy as np
from operator import neg

def load_data(file_path):
    """
    Load data from file where:
    - -1 separates events
    - -2 separates sequences
    - Other numbers are event IDs
    """
    sequences = []
    current_sequence = []
    current_event = set()
    
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                try:
                    num = int(token)
                    if num == -2:  # End of sequence
                        # if current_event:
                        #     current_sequence.append(sorted(current_event))
                        #     current_event = set()
                        if current_sequence:
                            sequences.append(current_sequence)
                            current_sequence = []
                    elif num == -1:  # End of event
                        if current_event:
                            current_sequence.append(sorted(current_event))
                            current_event = set()
                    else:  # Event ID
                        current_event.add(num)
                except ValueError:
                    # Skip non-integer tokens
                    continue
            
            # Handle end of line (treat as end of sequence)
            if current_event:
                current_sequence.append(sorted(current_event))
                current_event = set()
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
    
    return sequences

def is_subsequence(main_sequence, subsequence):
    """Check if subsequence is contained in main_sequence"""
    def is_subsequence_recursive(subsequence_clone, start=0):
        if not subsequence_clone:
            return True
        first_elem = set(subsequence_clone.pop(0))
        for i in range(start, len(main_sequence)):
            if set(main_sequence[i]).issuperset(first_elem):
                return is_subsequence_recursive(subsequence_clone, i + 1)
        return False
    return is_subsequence_recursive(copy.deepcopy(subsequence))

def sequence_length(sequence):
    """Calculate the total number of items in the sequence"""
    return sum(len(i) for i in sequence)

def count_support(file_path, cand_seq):
    """Count how many sequences contain the candidate sequence"""
    sequences = load_data(file_path)
    return sum(1 for seq in sequences if is_subsequence(seq, cand_seq))

def gen_cands_for_pair(cand1, cand2):
    """Generate a new candidate from two candidates of length k-1"""
    cand1_clone = copy.deepcopy(cand1)
    cand2_clone = copy.deepcopy(cand2)
    
    if len(cand1[0]) == 1:
        cand1_clone.pop(0)
    else:
        cand1_clone[0] = cand1_clone[0][1:]
        
    if len(cand2[-1]) == 1:
        cand2_clone.pop(-1)
    else:
        cand2_clone[-1] = cand2_clone[-1][:-1]
        
    if not cand1_clone == cand2_clone:
        return []
    else:
        new_cand = copy.deepcopy(cand1)
        if len(cand2[-1]) == 1:
            new_cand.append(cand2[-1])
        else:
            new_cand[-1].extend([cand2[-1][-1]])
        return new_cand

def gen_cands(last_lvl_cands):
    """Generate candidate sequences of length k+1 from frequent sequences of length k"""
    k = sequence_length(last_lvl_cands[0]) + 1
    
    if k == 2:
        flat_short_cands = [item for sublist2 in last_lvl_cands 
                          for sublist1 in sublist2 
                          for item in sublist1]
        result = [[[a, b]] for a in flat_short_cands 
                          for b in flat_short_cands 
                          if b > a]
        result.extend([[[a], [b]] for a in flat_short_cands 
                                     for b in flat_short_cands])
        return result
    else:
        cands = []
        for i in range(len(last_lvl_cands)):
            for j in range(len(last_lvl_cands)):
                new_cand = gen_cands_for_pair(last_lvl_cands[i], last_lvl_cands[j])
                if new_cand:
                    cands.append(new_cand)
        cands.sort()
        return cands

def gen_direct_subsequences(sequence):
    """Generate all possible direct subsequences of length k-1"""
    result = []
    for i, itemset in enumerate(sequence):
        if len(itemset) == 1:
            seq_clone = copy.deepcopy(sequence)
            seq_clone.pop(i)
            result.append(seq_clone)
        else:
            for j in range(len(itemset)):
                seq_clone = copy.deepcopy(sequence)
                seq_clone[i].pop(j)
                result.append(seq_clone)
    return result

def prune_cands(last_lvl_cands, cands_gen):
    """Prune candidates that have any (k-1)-subsequence that is not frequent"""
    return [cand for cand in cands_gen 
            if all(any(cand_subseq == freq_seq for freq_seq in last_lvl_cands) 
                 for cand_subseq in gen_direct_subsequences(cand))]

def gsp(dataset, file_path, min_sup, verbose=False):
    """
    GSP (Generalized Sequential Pattern) algorithm
    
    Parameters:
    - dataset: List of sequences (each sequence is a list of itemsets)
    - min_sup: Minimum support (fraction or absolute count)
    - verbose: Print progress if True
    
    Returns:
    - List of (sequence, support) tuples, sorted by support (descending)
    """
    # Convert min_sup to absolute count if it's a fraction
    if 0 < min_sup < 1:
        min_sup = int(min_sup * len(dataset))
    
    # Initialize
    overall = []
    
    # Get all unique items
    items = sorted(set([item for sequence in dataset
                       for itemset in sequence
                       for item in itemset]))
    
    # Generate 1-sequences
    single_item_sequences = [[[item]] for item in items]
    single_item_counts = [(s, count_support(file_path, s)) 
                         for s in single_item_sequences]
    single_item_counts = [(i, count) for i, count in single_item_counts 
                         if count >= min_sup]
    overall.append(single_item_counts)
    
    # Generate k-sequences for k > 1
    k = 1
    while overall[k - 1]:
        last_lvl_cands = [x[0] for x in overall[k - 1]]
        cands_gen = gen_cands(last_lvl_cands)
        cands_pruned = prune_cands(last_lvl_cands, cands_gen)
        cands_counts = [(s, count_support(file_path, s)) for s in cands_pruned]
        result_lvl = [(i, count) for i, count in cands_counts if count >= min_sup]
        
        if verbose > 1:
            print('Candidates generated, lvl', k + 1, ':', cands_gen)
            print('\nCandidates pruned, lvl', k + 1, ':', cands_pruned)
            print('Result, Level', k + 1, ':', result_lvl)
            print('-' * 100)
        
        overall.append(result_lvl)
        k += 1
    
    # Flatten the results and sort by support (descending)
    overall = overall[:-1]  # Remove empty last level
    overall = [item for sublist in overall for item in sublist]
    overall.sort(key=lambda tup: (tup[1], -sequence_length(tup[0])), reverse=True)
    
    return overall

def format_sequence(sequence):
    """Format a sequence for display"""
    return str(sequence).replace('], [', ' -> ').replace('[', '').replace(']', '')

# Example usage:
if __name__ == "__main__":

    file_path = r"Q:\Y3S1\4Y-S1\CSE-4225 Data Mining & Warehousing\Labworks\Zisan\b.txt"
    sequences = load_data(file_path)
    print(f"Loaded {len(sequences)} sequences from file")
    
    # Run GSP
    min_sup = 0.5  # 50% support
    print(f"Running GSP with min_sup={min_sup}")
    results = gsp(sequences, file_path, min_sup, verbose=True)
    
    # Print results
    print("\nFrequent sequences:")
    for i, (seq, support) in enumerate(results[:54], 1):
        print(f"{i}. {format_sequence(seq)} (Support: {support})")
    
    print(f"\nTotal frequent sequences found: {len(results)}")