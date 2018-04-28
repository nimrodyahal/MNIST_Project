# -*- coding: utf-8 -*-
import SimpleHTTPServer
import SocketServer
import os
import string
import re
import threading
import cv2
from Queue import Queue
from preprocessor import Preprocessor
from memoized_decorator import Memoized


PORT = 80
SAVE_DIR = 'Cache\\'
FILE_NAME_FORMAT = 'img{number}.png'


def _get_file_path():
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


def _set_path_free(path):
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


class HttpServerThread(threading.Thread):
    def __init__(self, input_queue, spell_checker):
        global used_numbers
        used_numbers = []
        threading.Thread.__init__(self)
        # input_queue = Queue.Queue()
        self.port = PORT
        self.httpd = MyTCPServer(('', self.port), MyHandler, input_queue,
                                 spell_checker)
        # self.httpd = SocketServer.TCPServer(('', self.port), MyHandler)

    def run(self):
        print 'HTTP Server: Stating Up'
        self.httpd.serve_forever()
        print 'HTTP Server: Shutting Down'


class MyTCPServer(SocketServer.TCPServer):
    def __init__(self, server_address, RequestHandlerClass, input_queue,
                 spell_checker):
        SocketServer.TCPServer.__init__(self, server_address,
                                        RequestHandlerClass)
        self.input_queue = input_queue
        self.spell_checker = spell_checker

    def finish_request(self, request, client_address):
        self.RequestHandlerClass(request, client_address,
                                 self, self.input_queue, self.spell_checker)


class MyHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def __init__(self, request, client_address, server, input_queue,
                 spell_checker):
        self.input_queue = input_queue
        self.output_queue = Queue()
        self.spell_checker = spell_checker
        SimpleHTTPServer.SimpleHTTPRequestHandler.__init__(self, request,
                                                           client_address,
                                                           server)

    def __get_file_name(self):
        request = self.requestline
        code_word_start = 'file-name='
        code_word_end = ' HTTP/'

        index_start = request.index(code_word_start) + len(code_word_start)
        index_end = request.index(code_word_end)
        raw_name = request[index_start:index_end]

        split_name = raw_name.split('%')
        for i, char in enumerate(split_name[1:]):
            if all(c in string.hexdigits for c in char[:2]):
                char = chr(int(char[:2], 16)) + char[2:]
            split_name[1 + i] = char
        name = ''.join(split_name)
        if not re.findall(r'[^A-Za-z0-9_\-\. ]', name):
            return name
        return False

    def do_GET(self):
        self.path = 'http/' + self.path
        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        file_data = self.rfile.read(length)
        # filename = self.__get_file_name()
        # if filename:
        answer = self.classify(file_data)
        # with open(filename, 'wb') as f:
        #     f.write(file_data)
        self.send_response(201)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        self.wfile.write(
            "<html><body><h1>{}</h1></body></html>".format(answer))
        # else:
        #     pass
        # self.input_queue.put_nowait(file_data)

    @Memoized
    def classify(self, file_data):
        cached_img_path = _get_file_path()
        with open(cached_img_path, 'wb') as img_file:
            img_file.write(file_data)

        cv2_img = cv2.imread(cached_img_path, 0)
        _set_path_free(cached_img_path)
        preprocessor = Preprocessor(cv2_img)
        separated = preprocessor.separate_text()
        print 'HTTP Request Handler: Separated Chars'
        self.input_queue.put_nowait((separated, self.output_queue))
        string_text = self.wait_for_answer()
        print 'HTTP Request Handler: Classification Done!'
        auto_completed = self.spell_checker.autocomplete_text(string_text)
        print 'HTTP Request Handler: Spell Checking Done!'
        print auto_completed
        return auto_completed

    def wait_for_answer(self):
        while True:
            if not self.output_queue.empty():
                return self.output_queue.get_nowait()