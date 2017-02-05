import numpy as np
import pandas as pd

prefix='tree'

print('Reading count')
count = np.array(pd.read_csv('%s.count'%prefix, delimiter='\t', header=None))[:, :-1]
np.save('%s.np_count' % prefix, count)
