import sys
import os

current_file = os.path.abspath(__file__)

parent_dir = os.path.dirname(os.path.dirname(current_file))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
