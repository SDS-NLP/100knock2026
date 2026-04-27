#!/bin/bash
FILE="popular-names.txt"
N=10

LINES=$(wc -l <"$FILE")
LINES_PER_FILE=$(((LINES + N - 1) / N))

split -l "$LINES_PER_FILE" "$FILE"
