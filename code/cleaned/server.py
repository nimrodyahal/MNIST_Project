import socket
import threading
import os
from pre_processing import preprocess_img
from handle_nn import load_multi_net


SAVE_DIR = 'Cache\\'
FILE_NAME_FORMAT = 'img{number}.png'


def get_file_path():
    """
    Returns a free file name path. The names are by the convention of
    FILE_NAME_FORMAT with the {number} changing based on what is free. By
    free, I mean if the program stopped using a file name, it would designate
    it as 'free'. If there are no free names, it will make another number by
    just taking the highest number and increasing it by 1.
    :return: The full path to be used when creating a new image file.
    """
    global used_numbers
    if not used_numbers:
        name = 0
    else:
        used_numbers = sorted(used_numbers)
        full_range = range(used_numbers[-1])
        unused_names = list(set(full_range) - set(used_numbers))  # Check for
        # any unused names within the range of names already created
        if unused_names:
            name = unused_names[0]
        else:  # Expand the range by created another name
            name = used_numbers[-1] + 1
    used_numbers.append(name)
    return SAVE_DIR + FILE_NAME_FORMAT.format(number=name)


def set_path_free(path):
    """
    Designates a file name 'free' and deletes the file with that name if it
    exists, meaning it will be reused. It is important to use this function
    once you finish using the file, otherwise more and more files will be
    created and none deleted.
    :param path:
    """
    global used_numbers
    os.remove(path)
    index_of_number = (SAVE_DIR + FILE_NAME_FORMAT).index('{')
    number_of_path = int(path[index_of_number])
    used_numbers.remove(number_of_path)


class ClientConnectionThread(threading.Thread):
    def __init__(self, server, conn_success, multi_net):
        threading.Thread.__init__(self)
        self.__multi_net = multi_net
        self.__server = server
        self.__conn, _ = self.__server.accept()
        conn_success.set()  # Sets an event for a successful connection

    def run(self):
        while True:
            img = self.__conn.recv(512)
            self.__conn.send(self.classify(img))
        self.__conn.close()

    def classify(self, img):
        file_path = get_file_path()
        with open(file_path, 'wb') as img_file:
            img_file.write(img)
        char = preprocess_img(file_path)
        classifications = self.__multi_net.feedforward(char)
        set_path_free(file_path)
        return classifications


def main():
    global used_numbers
    used_numbers = []
    client_threads = []
    multi_net = load_multi_net(['..\\Saved Nets\\test_net0.txt'])

    server = socket.socket()
    server.bind(('0.0.0.0', 500))
    server.listen(10)

    conn_success = threading.Event()
    new_client = ClientConnectionThread(server, conn_success, multi_net)
    new_client.start()
    while True:
        if conn_success.is_set():  # If all threads are already connected to a
        #  client, create a new one
            conn_success.clear()
            client_threads.append(new_client)
            new_client = ClientConnectionThread(server, conn_success,
                                                multi_net)
            new_client.start()


if __name__ == '__main__':
    main()