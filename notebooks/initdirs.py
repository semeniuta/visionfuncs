import sys
import os

CODE_DIR = os.environ['PHD_CODE']
ROOT_DIR = os.path.abspath('..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

sys.path.append(os.path.join(CODE_DIR, 'EPypes'))
sys.path.append(ROOT_DIR)