import sys
sys.path.append("../src")

from batchmanager import BatchManager
import numpy as np

bm = BatchManager(20, 5)
#sys.exit()
for i in bm:
    print(i)
