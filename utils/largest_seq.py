import csv
import sys

csv.field_size_limit(sys.maxsize)

def longest_sequence_length(csv_filename):
    max_length = 0

    with open(csv_filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            sequence_1_length = len(row[1])
            sequence_2_length = len(row[2])
            max_length = max(max_length, sequence_1_length, sequence_2_length)

    return max_length

if __name__ == "__main__":
    csv_filename = input("Enter the CSV file name: ")
    longest_seq_length = longest_sequence_length(csv_filename)
    print("Length of the longest sequence:", longest_seq_length)