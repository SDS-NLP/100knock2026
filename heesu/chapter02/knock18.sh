#!/bin/bash

cut -f 1 popular-names.txt | sort | uniq -c | sort -gr | awk '{print $2}'
