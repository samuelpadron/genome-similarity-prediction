import sys
import csv
from tqdm import tqdm #progress bars

def parse_maf(maf_file, output_csv):
    with open(maf_file, 'r') as file_in, open(output_csv, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(['pair_id', 'blastz_score', 'sequence_1', 'sequence_2'])

        sequences = []
        pair_id = 1
        blastz_score = None
        for line in tqdm(file_in):
            if line.startswith('a'):
                blastz_score = float(line.strip().split()[1].split('=')[1])
            elif line.startswith('s'):
                sequences.append(line.strip().split()[-1].replace('-', ''))

                if len(sequences) == 2:
                    sequence_1, sequence_2 = sequences
                    writer.writerow([f'pair_true_{pair_id}', blastz_score, sequence_1, sequence_2])
                    sequences = []
                    pair_id += 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <MAF file> <output CSV>")
        sys.exit(1)
    maf_file = sys.argv[1] 
    output_csv = sys.argv[2] 

    parse_maf(maf_file, output_csv)
