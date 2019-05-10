import ctypes

connect_lib = ctypes.CDLL("./final_server.so")
init_conn = connect_lib.init_serial
stop_conn = connect_lib.end_serial
connect_lib.send_facial_pos.argtypes = [ctypes.c_double, ctypes.c_double]
send_msg = connect_lib.send_facial_pos

# Try to modify set point for fixed times
_ = init_conn()

