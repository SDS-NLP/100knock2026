#!/usr/bin/env bash
set -euo pipefail

input_file="jawiki-country.json"
output_file="uk.txt"

jq -r 'select(.title == "イギリス") | .text' "$input_file" > "$output_file"

echo "Created $output_file from $input_file"
