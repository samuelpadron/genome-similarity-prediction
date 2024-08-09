import sys
import csv
from tqdm import tqdm

def parse_maf(maf_file, output_csv):
    max_length = 5000
    max_sequences = 16000
    
    with open(maf_file, 'r') as file_in, open(output_csv, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(['pair_id', 'blastz_score', 'sequence_1', 'sequence_2'])

        sequences = []
        pair_id = 1
        blastz_score = None
        for line in tqdm(file_in):
            if pair_id > max_sequences:
                break
            if line.startswith('a'):
                blastz_score = float(line.strip().split()[1].split('=')[1])
            elif line.startswith('s'):
                sequences.append(line.strip().split()[-1].replace('-', ''))

                if len(sequences) == 2:
                    sequence_1, sequence_2 = sequences
                    
                    while len(sequence_1) > max_length:
                        if pair_id > max_sequences:
                            break
                        writer.writerow([f'pair_true_{pair_id}', blastz_score, sequence_1[:max_length], sequence_2[:max_length]])
                        pair_id += 1
                        sequence_1 = sequence_1[max_length:]
                        sequence_2 = sequence_2[max_length:]
                    
                    # don't write pairs with no seq2
                    if pair_id <= max_sequences and len(sequence_2) != 0:
                        writer.writerow([f'pair_true_{pair_id}', blastz_score, sequence_1, sequence_2])
            
                        pair_id += 1
                        
                    sequences = []

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <MAF file> <output CSV>")
        sys.exit(1)
    maf_file = sys.argv[1]
    output_csv = sys.argv[2]

    parse_maf(maf_file, output_csv)