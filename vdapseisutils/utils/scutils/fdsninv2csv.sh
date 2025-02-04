#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.txt output.csv"
    exit 1
fi

# Input and output file names from arguments
input_file="$1"
output_file="$2"

# Write the header to the output file
echo "# stationID, latitude, longitude, elevation, depth" > "$output_file"

# Process the input file and append to the output file
awk -F'|' 'NR > 1 { 
    stationID = $1 "." $2 "." $3 "." $4; 
    print stationID "," $5 "," $6 "," $7 "," $8 
}' "$input_file" >> "$output_file"

echo "Conversion complete. Output saved to $output_file."

