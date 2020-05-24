import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import Xlib
import Xlib.display
try:
    disp = Xlib.display.Display()
except Exception:
    pass

def isFocussed():
    window = disp.get_input_focus().focus
    return type(window) is not int and sys.argv[0] == window.get_wm_name()

def saveNet(net, filename):
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump(net, f, protocol=pickle.HIGHEST_PROTOCOL)
