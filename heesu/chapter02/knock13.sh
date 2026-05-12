#!/bin/bash
head -n 10 popular-names.txt | sed 's/\t/ /g'
