import random
import pandas as pd
from tqdm import tqdm

def generate_random_dna_sequence(length):
    bases = ['A', 'C', 'G', 'T']
    return ''.join(random.choices(bases, k=length))

num_pairs = 100
max_length = 500

data = []

for i in tqdm(range(1, num_pairs + 1)):
    sequence = generate_random_dna_sequence(max_length)
    
    pair_data = {
        'sequence_1': sequence,
        'sequence_2': sequence,
        'label': 1
    }
    
    data.append(pair_data)

df = pd.DataFrame(data)
df.to_csv('/vol/csedu-nobackup/project/spadronalcala/simulated_dna_pairs.csv', index=True)