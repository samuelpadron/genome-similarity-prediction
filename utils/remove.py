import csv

def remove_dashes_from_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        rows = list(reader)
        
    cleaned_rows = []
    for row in rows:
        cleaned_row = [field.replace('-', '') for field in row]
        cleaned_rows.append(cleaned_row)
        
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)

# Example usage
input_csv = '/vol/csedu-nobackup/project/spadronalcala/pair_alignment/danRer10_true.csv'
output_csv = '/vol/csedu-nobackup/project/spadronalcala/pair_alignment/danRer10_true.csv'
remove_dashes_from_csv(input_csv, output_csv)
