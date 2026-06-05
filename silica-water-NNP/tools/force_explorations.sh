#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2024/05/26
# Last modified: 2024/06/27
#----------------------------------------------------------------------------------------------------#

# Check if the input file exists
input_file="failed_explorations.txt"
if [[ ! -f "$input_file" ]]; then
    echo "Error: File '$input_file' not found!"
    exit 1
fi

# Read the input file line by line
while IFS= read -r directory; do
    # Trim leading and trailing whitespaces
    directory=$(echo "$directory" | xargs)

    # Check if the directory is not empty
    if [[ -n "$directory" ]]; then
        # Create the "force" file in the directory
        touch "$directory/force" && echo "Created force file in: $directory"
    else
        echo "Warning: Empty line encountered in the input file."
    fi
done < "$input_file"

echo "Script execution completed."