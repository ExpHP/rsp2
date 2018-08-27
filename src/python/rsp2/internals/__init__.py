import sys
import time

# (you can add calls to this if you want to do some "println profiling".
#  rsp2 will timestamp the lines as they are received)
def info(*args):
    print(*args, file=sys.stderr); sys.stderr.flush(); time.sleep(0)
