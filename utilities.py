import sys
import Xlib
import Xlib.display
disp = Xlib.display.Display()

def isFocussed():
    window = disp.get_input_focus().focus
    return type(window) is not int and sys.argv[0] == window.get_wm_name()


