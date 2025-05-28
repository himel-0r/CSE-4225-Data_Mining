import copy
import time
from memory_profiler import memory_usage
from math import ceil
import os

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

def is_subsequence(main_seq, subsequence):
    def is_subsequences_recursion(subsequence_clone, start = 0):
        if not subsequence_clone:
            return True
        
        first_element = set(subsequence_clone.pop(0))
        for i in range(start, len(main_seq)):
            if set(main_seq[i]).issuperset(first_element):
                return is_subsequences_recursion(subsequence_clone, i+1)
        return False
    return is_subsequences_recursion(copy.deepcopy(subsequence))

def count_support(filepath, candidate_sequence):
    sequences = load_data(filepath)
    return sum(1 for seq in sequences if is_subsequence(seq, candidate_sequence))

def sequence_length(sequence):
    return sum(len(i) for i in sequence)
    
def gen_cadidate_pair(candidate1, candidate2):
    candi1 = copy.deepcopy(candidate1)
    candi2 = copy.deepcopy(candidate2)
    
    if len(candi1[0]) == 1:
        candi1.pop(0)
    else:
        candi1[0] = candi1[0][1:]
    
    if len(candi2[-1]) == 1:
        candi2.pop(-1)
    else:
        candi2[-1] = candi2[-1][:-1]
        
    if not candi1 == candi2:
        return []
    else:
        new_candi = copy.deepcopy(candi1)
        if len(candi2[-1]) == 1:
            new_candi.append(candi2[-1])
        else:
            new_candi[-1].extend([candi2[-1][-1]])
        return new_candi
    
def generate_candidates(last_level_candidates):
    k = sequence_length(last_level_candidates[0]) + 1
    
    if k == 2:
        flat_short_candidates = [item for sublist2 in last_level_candidates 
                                 for sublist1 in sublist2 
                                 for item in sublist1]
        result = [[[a, b]] for a in flat_short_candidates
                  for b in flat_short_candidates
                  if b > a]
        result.extend([[[a], [b]] for a in flat_short_candidates
                       for b in flat_short_candidates])
        return result
    else:
        candidates = []
        for i in range(len(last_level_candidates)):
            for j in range(len(last_level_candidates)):
                new_candidate = gen_cadidate_pair(last_level_candidates[i], last_level_candidates[j])
                if new_candidate:
                    candidates.append(new_candidate)
        return candidates
    
def generate_direct_subsequences(sequence):
    #k-1 length er all possible direct subsequences generate kora hobe
    result = []
    for i, itemset in enumerate(sequence):
        if len(itemset) == 1:
            sequence_clone = copy.deepcopy(sequence)
            sequence_clone.pop(i)
            result.append(sequence_clone)
        else:
            for j in range(len(itemset)):
                sequence_clone = copy.deepcopy(sequence)
                sequence_clone[i].pop(j)
                result.append(sequence_clone)
        return result
    
def prune_candidates(last_level_candidates, candidates_gen):
    # non-frequent subsequence thaka candidate gula prune korbo
    return [cand for cand in candidates_gen
            if all(any(cand_subseq == freq_seq for freq_seq in last_level_candidates)
                   for cand_subseq in generate_direct_subsequences(cand))]
    
def gsp_printing(dataset, min_sup, verbose, output_file="gsp_outputs.txt"):
    file_out = open(output_file, 'w') if output_file else None
    
    def print_and_write(text):
        print(text)
        if file_out:
            file_out.write(text + "\n")
    
    if 0 < min_sup < 1:
        min_sup = int(min_sup * len(dataset))
        
    overall = []
    
    items = sorted(set([item for sequence in dataset
                        for itemset in sequence
                        for item in itemset]))
    
    single_item_sequences = [[[item]] for item in items]
    single_item_counts = [(s, count_support(dataset, s)) 
                         for s in single_item_sequences]
    single_item_counts = [(i, count) for i, count in single_item_counts 
                         if count >= min_sup]
    overall.append(single_item_counts)
    
    k = 1
    while overall[k-1]:
        last_lvl_cands = [x[0] for x in overall[k - 1]]
        cands_gen = generate_candidates(last_lvl_cands)
        cands_pruned = prune_candidates(last_lvl_cands, cands_gen)
        cands_counts = [(s, count_support(dataset, s)) for s in cands_pruned]
        result_lvl = [(i, count) for i, count in cands_counts if count >= min_sup]
        
        overall.append(result_lvl)
        k += 1
    
    # Flatten the results and sort by support (descending)
    overall = overall[:-1]  # Remove empty last level
    overall = [item for sublist in overall for item in sublist]
    overall.sort(key=lambda tup: (tup[1], -sequence_length(tup[0])), reverse=True)
    
    if file_out:
        file_out.close()
        print("Results saved")
    
    return overall

def gsp(dataset, min_sup, verbose):
    start_time = time.time()
    start_memory = memory_usage()[0]
    
    if 0 < min_sup < 1:
        min_sup = int(ceil(min_sup * len(dataset)))
        
    overall = []
    
    items = sorted(set([item for sequence in dataset
                       for itemset in sequence
                       for item in itemset]))
    
    single_item_sequences = [[[item]] for item in items]
    single_item_counts = [(s, count_support(dataset, s)) 
                         for s in single_item_sequences]
    single_item_counts = [(i, count) for i, count in single_item_counts 
                         if count >= min_sup]
    overall.append(single_item_counts)
    
    k = 1
    while overall[k - 1]:
        last_lvl_cands = [x[0] for x in overall[k - 1]]
        cands_gen = generate_candidates(last_lvl_cands)
        cands_pruned = prune_candidates(last_lvl_cands, cands_gen)
        cands_counts = [(s, count_support(dataset, s)) for s in cands_pruned]
        result_lvl = [(i, count) for i, count in cands_counts if count >= min_sup]
        
        if verbose :
            print('Candidates generated, lvl', k + 1, ':', cands_gen)
            print('\nCandidates pruned, lvl', k + 1, ':', cands_pruned)
            print('Result, Level', k + 1, ':', result_lvl)
            print('-' * 100)
        
        overall.append(result_lvl)
        k += 1
    
    overall = overall[:-1]  # Remove empty last level
    overall = [item for sublist in overall for item in sublist]
    overall.sort(key=lambda tup: (tup[1], -sequence_length(tup[0])), reverse=True)
    
    end_time = time.time()
    end_memory = memory_usage()[0]
    memory_consumption = end_memory - start_memory
    execution_time = end_time - start_time
    
    return overall, memory_consumption, execution_time


def main(filename, min_sup):
    file_path_name = os.path.join(os.getcwd(), "DATASET")
    
    file_path = file_path_name + r"\\" + filename
    sequences = load_data(file_path)
    
    print(f"Loaded {len(sequences)} sequences from {filename}")
    print(f"Running GSP with min_sup={min_sup}")
    
    results, memory_usages, execution_time = gsp(sequences, min_sup, verbose=False)
    total_frequent_items = len(results)
    
    print(f"(Min_Sup = {min_sup})")
    print(f"Total memory used by algorithm: {memory_usages:.2f} MB")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total number of frequent items: {total_frequent_items}")
    
    O_file_path = os.path.join(os.getcwd(), "Output")
    O_file_name = O_file_path + r"\\" + filename
    
    with open(O_file_name, "a") as f:
        f.write(f"(Min_Sup = {min_sup})\n")
        f.write(f"Total memory used by algorithm: {memory_usages:.4f} MB\n")
        f.write(f"Total execution time: {execution_time:.4f} seconds\n")
        f.write(f"Total number of frequent items: {total_frequent_items}\n")

def ESHOP():
    filename = r"e_shop.txt"
    sups = [0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.75]
    
    for sup in sups:
        main(filename, sup)

def BMS1():
    filename = r"BMS1_spmf.txt"
    sups = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    
    for sup in sups:
        main(filename, sup)
        
def BIKE():
    filename = r"BIKE.txt"
    sups = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    
    for sup in sups:
        main(filename, sup)
        
def SIGN():
    filename = r"SIGN.txt"
    sups = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    
    for sup in sups:
        main(filename, sup)
    

if __name__ == "__main__":
    filename = r"book.txt"
    main(filename, 0.5)
    
    # ESHOP()
    # BMS1()
    # BIKE()
    # SIGN()