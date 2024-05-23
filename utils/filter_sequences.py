import csv

def filter_sequences(input_filename, output_filename):
    with open(input_filename, 'r') as input_file, open(output_filename, 'w', newline='') as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)
        
        header = next(csv_reader)
        csv_writer.writerow(header)
        
        for row in csv_reader:
            sequence_1_len = len(row[2])
            sequence_2_len = len(row[3])
            
            if sequence_1_len >= 1024 and sequence_2_len >= 1024:
                csv_writer.writerow(row)
                
if __name__ == "__main__":
    input_filename = input("Enter input filename:")
    output_filename = input("Enter output filename:")
    
    filter_sequences(input_filename, output_filename)
        