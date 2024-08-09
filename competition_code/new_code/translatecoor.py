import os

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()

if script_dir != cwd:
    print("Warning: The script is not running from its own directory.")
    print(f"Script Directory: {script_dir}")
    print(f"Current Working Directory: {cwd}")

file_path = os.path.join(script_dir, "waypoints_new.txt")

def convert_coordinates(input_filename, output_filename):
    # Read the input file
    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    # Process the lines to convert them to the desired format
    formatted_lines = []
    for line in lines:
        # Remove any leading/trailing whitespace and split the line by comma
        x, y = line.strip().split(',')
        # Format the coordinates and add them to the list
        formatted_lines.append(f"({x.strip()}, {y.strip()}),")

    # Write the formatted lines to the output file
    with open(output_filename, 'w') as outfile:
        outfile.write('\n'.join(formatted_lines))

# Example usage
input_filename = 'waypoints_new.txt'
output_filename = 'output.txt'
convert_coordinates(input_filename, output_filename)