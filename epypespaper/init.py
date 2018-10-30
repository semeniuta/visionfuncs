import sys
import os

CODE_DIR = os.environ['PHD_CODE']

def init():

    paths = (os.path.join(CODE_DIR, p) for p in ('EPypes', 'RPALib'))

    for p in paths:
        if not p in sys.path:
            sys.path.append(p)

init()