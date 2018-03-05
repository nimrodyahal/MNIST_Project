import socket
import threading


def get_file_name():
    global names
    save_dir = 'Cache\\'
    if not names:
        name = 0
    else:
        names = list(set(names))
        full_range = range(names[-1])
        unused_names = list(set(full_range) - set(names))
        if unused_names:
            name = unused_names[0]
        else:
            name = names[-1] + 1
    names.append(name)
    return save_dir + 'img{}.txt'.format(name)


class ClientConnectionThread(threading.Thread):
    def __init__(self, server, conn_success):
        threading.Thread.__init__(self)
        self.__server = server
        self.__conn, adrr = self.__server.accept()
        conn_success.set()  # Sets an event for a successful connection

    def run(self):
        img = self.__conn.recv(512)
        with open('bla.png', 'wb') as img_file:
            img_file.write(img)
        #######################################
        self.__conn.close()


def main():
    global names
    names = []
    seed = ''  # The decoded answer
    client_threads = []

    server = socket.socket()
    server.bind(('0.0.0.0', 500))
    server.listen(10)

    conn_success = threading.Event()
    new_client = ClientConnectionThread(server, conn_success)
    new_client.start()
    while not seed:  # Stops once the answer is found
        if conn_success.is_set():  # If all threads are already connected to a
        #  client, create a new one
            conn_success.clear()
            client_threads.append(new_client)
            new_client = ClientConnectionThread(server, conn_success)
            new_client.start()
    print seed


if __name__ == '__main__':
    main()