import sys
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))
HOME = '{}/..'

sys.path.append('HOME'.format(this_file_path))
