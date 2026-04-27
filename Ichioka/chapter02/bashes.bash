# q10
wc -l popular-names.txt

# q11
head -n 10 popular-names.txt

# q12
tail -n 10 popular-names.txt

# q13
sed '1,10 s/\t/ /g' popular-names.txt

# q14
head -n 10 popular-names.txt | cut -d ' ' -f1

# q15
split -n 10 popular-names.txt out_

# q16
shuf popular-names.txt

# q17
cut -d ' ' -f1 popular-names.txt | sort | uniq

# q18
awk '{print $1}' popular-names.txt | sort | uniq -c | sort -nr

# q19
sort -k3,3 -nr popular-names.txt