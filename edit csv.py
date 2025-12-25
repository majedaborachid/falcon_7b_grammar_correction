import csv

# Input and output file paths
input_csv_path = r'C:\DATA\study\NN project\dataset3.csv'
output_csv_path = r'C:\DATA\study\NN project\my_grammar1.csv'

# Text to add in the middle of each line
text_to_add = 'correct the following sentence: '

# Read the CSV file, add the text in the middle, and write to a new CSV file
with open(input_csv_path, 'r', newline='') as input_file, open(output_csv_path, 'w', newline='') as output_file:
    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(output_file)

    for row in csv_reader:
        # Add the text in the middle of each line (after the second column)
        row[1]=f'{text_to_add}{row[1]}'
        modified_row = row[1:]
        # Write the modified row to the new CSV file
        csv_writer.writerow(modified_row)

print(f"Text '{text_to_add}' added in the middle of each line. Output saved to {output_csv_path}.")
