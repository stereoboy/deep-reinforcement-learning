import numpy as np
import matplotlib.pyplot as plt
import sys

filelist = sys.argv[1:]
scoreslist = []
for filepath in filelist:
    scoreslist.append(np.load(filepath))

## plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
for scores in scoreslist:
    plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
