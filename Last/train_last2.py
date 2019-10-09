import os
import sys

iteration = 50

if len(sys.argv) > 1:
    iteration = int(sys.argv[1])

if iteration == 0:
    os.system('python3 last2.py 0 {}'.format(iteration))
else:
    os.system('python3 last2.py 1 {}'.format(iteration))
    os.system('python3 last2.py 2 {}'.format(iteration))
    os.system('python3 last2.py 3 {}'.format(iteration))
    os.system('python3 last2.py 4 {}'.format(iteration))
    os.system('python3 last2.py 5 {}'.format(iteration))
    os.system('python3 last2.py 6 {}'.format(iteration))

