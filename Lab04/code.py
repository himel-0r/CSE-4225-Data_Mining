import copy

def load_data(file_path):
    sequences = []
    current_sequence = []
    current_event = set()
    
    """ -1 diye event separate, -2 diye transaction separate
    """

    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            for token in tokens:
                number = int(token)
                
                if number == -2:
                    if current_event:
                        current_sequence.append(current_event)
                        current_event = set()
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
    
def join_candidates(candidate1, candidate2):
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
        return