from time import sleep
from random import random
import ctypes

from math import pi, sin, cos

Fs=8000
f=500
sample=16
a=[0]*sample
for n in range(sample):
    a[n]=sin(2*pi*f*n/Fs)

connect_lib = ctypes.CDLL("./final_server.so")
init_conn = connect_lib.init_serial
stop_conn = connect_lib.end_serial
connect_lib.send_facial_pos.argtypes = [ctypes.c_double, ctypes.c_double]
send_msg = connect_lib.send_facial_pos

# Try to modify set point for fixed times
_ = init_conn()

Fs = 100
f = 20
amp = 0.5 * 90.0

for rd in range(4):
    print("Start round: {:d}".format(rd + 1))
    for i in range(30):
        # Generate value between [-90 ~ 90]
        """
        rand_x, rand_y = (random() - 0.5) * 180.0, (random() - 0.5) * 180.0
        print("Try to modify position for position ({:f}, {:f})...".format(rand_x, rand_y))
        send_msg(rand_x, rand_y)
        """
        angle = 2 * pi * f * i / Fs
        x_pos, y_pos = amp * sin(angle), amp * cos(angle)
        print("Try to modify position for position ({:f}, {:f})...".format(x_pos, y_pos))
        send_msg(x_pos, y_pos) 
        sleep(0.45)
    send_msg(180.0, 180.0)
    sleep(3)
_ = stop_conn()
print("All set!")
