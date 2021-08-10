import socket

import numpy as np


def recvall(sock: socket, count: int) -> bytes:
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return buf
        buf += newbuf
        count -= len(newbuf)
    return buf


class Sender:
    def __init__(self):
        self.client_socket: socket

    def send_and_recv(self, datas: dict) -> np.ndarray:
        self.client_socket = socket.socket(
            family=socket.AF_INET,
            type=socket.SOCK_STREAM
        )
        self.client_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )
        self.client_socket.connect(('127.0.0.1', 8080))
        datas = datas
        for key, item in datas.items():
            payload: bytes = item.tobytes()
            length: bytes = str(len(payload)).ljust(16).encode()
            print(f"sending {key} [size :{length}, shape :{item.shape} ]")
            self.client_socket.send(length)
            self.client_socket.send(payload)

        gen_length = recvall(self.client_socket, 16).decode()
        gen_payload = recvall(self.client_socket, int(gen_length))
        self.client_socket.close()
        self.client_socket = None

        buffered_data = np.frombuffer(gen_payload, dtype="uint8")
        return np.resize(buffered_data, (512, 512, 3))
