# -*- coding: utf-8 -*-
import threading
import Queue
from handle_nn import load_multi_net
from autocomplete import SpellChecker
from http_server import HttpServerThread


SAVE_DIR = 'Cache\\'
FILE_NAME_FORMAT = 'img{number}.png'


class NeuralNetThread(threading.Thread):
    """
    A thread that handles the Neural Network. It listens to input_queue.
    """
    def __init__(self, input_queue, multi_net):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.__multi_net = multi_net

    def run(self):
        print 'Neural Net Thread: Starting up'
        while True:
            try:
                separated, output_queue = self.input_queue.get_nowait()
                result = self.__multi_net.classify_text(separated)
                output_queue.put_nowait(result)
            except Queue.Empty:
                continue
        print 'Neural Net Thread: Shutting Down'


def main():
    multi_net = load_multi_net(['..\\Saved Nets\\test_net0.txt'])
    spell_checker = SpellChecker('big_merged.txt')
    input_queue = Queue.Queue()
    # Neural Network thread
    nn_thread = NeuralNetThread(input_queue, multi_net)
    nn_thread.start()
    # HTTP thread
    http_server = HttpServerThread(input_queue, spell_checker)
    http_server.start()

    # Wait until all threads are finished before closing the program
    nn_thread.join()
    http_server.join()


if __name__ == '__main__':
    main()