import sys
import csv
from tqdm import tqdm

def read_csv(input_file):
    pairs = []
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in tqdm(enumerate(reader)):
            pair_id = f"pair_false_{i+1}"
            pair = {'pair_id': pair_id, 'sequence_1': row['sequence_1'], 'sequence_2': row['sequence_2']}
            pairs.append(pair)
    return pairs

def shuffle_pairs(pairs):
    shuffled_pairs = pairs[:]
    for i in tqdm(range(len(shuffled_pairs) - 1)):
        shuffled_pairs[i]['sequence_1'], shuffled_pairs[i+1]['sequence_1'] = shuffled_pairs[i+1]['sequence_1'], shuffled_pairs[i]['sequence_1']
    return shuffled_pairs

def write_csv(pairs, output_file):
    with open(output_file, 'w', newline='') as file:
        fieldnames = pairs[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input CSV> <output CSV>")
        sys.exit(1)

    input_file = sys.argv[1] 
    output_file = sys.argv[2]
    
    pairs = read_csv(input_file)
    shuffled_pairs = shuffle_pairs(pairs)
    write_csv(shuffled_pairs, output_file)
