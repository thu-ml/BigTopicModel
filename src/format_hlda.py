#!/usr/bin/env python
import sys, shutil, random

i_prefix = sys.argv[1]
o_prefix = sys.argv[2]
parts = int(sys.argv[3])

# Copy vocab
shutil.copyfile(i_prefix + '.vocab', o_prefix + '.vocab')

# Read input
data = open(i_prefix + '.libsvm.train.0').readlines()
random.shuffle(data)
num_lines = len(data)
part_size = num_lines // parts + 1

for i in range(parts):
    start = i * part_size
    end = min(start + part_size, num_lines)

    with open(o_prefix + '.libsvm.train.' + str(i), 'w') as f:
        f.writelines(data[start:end])

