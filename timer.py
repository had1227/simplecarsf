import time
import datetime

start_time = time.time()

def start_timer():
    start_time = time.time()

def print_time():

    timer = str(datetime.timedelta(seconds=time.time() - start_time))
    print ("time: {}".format(timer))

    return timer

def current_time():

    timer = str(datetime.timedelta(seconds=time.time() - start_time))

    return timer
