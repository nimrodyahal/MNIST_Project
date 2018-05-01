# -*- coding: utf-8 -*-
import SimpleHTTPServer
import SocketServer
import numpy
import string
import re
import threading
import cv2
import cPickle
import urllib
from translate import translate
from Queue import Queue
from preprocessor import Preprocessor
from memoized_decorator import Memoized


PORT = 80
SAVE_DIR = 'Cache\\'
FILE_NAME_FORMAT = 'img{number}.png'
ADDRESS = 'mnistproject.ddns.net'
EXTERNAL_PORT = '500'
TRAN_LANG_PATH = 'Translation Languages.txt'  # The database containing all the
# languages Google can translate


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
        self.__cached_img_path = ''
        with open(TRAN_LANG_PATH, 'rb') as f:
            self.languages_dict = cPickle.load(f)
        SimpleHTTPServer.SimpleHTTPRequestHandler.__init__(self, request,
                                                           client_address,
                                                           server)

    def __get_file_name(self):
        request = self.requestline
        pattern = r'file-name=(.*) HTTP\/'
        raw_name = re.findall(pattern, request)[0]
        # code_word_start = 'file-name='
        # code_word_end = ' HTTP/'
        #
        # index_start = request.index(code_word_start) + len(code_word_start)
        # index_end = request.index(code_word_end)
        # raw_name = request[index_start:index_end]

        split_name = raw_name.split('%')
        for i, char in enumerate(split_name[1:]):
            if all(c in string.hexdigits for c in char[:2]):
                char = chr(int(char[:2], 16)) + char[2:]
            split_name[1 + i] = char
        name = ''.join(split_name)
        if not re.findall(r'[^A-Za-z0-9_\-\. ]', name):
            if '.' in name:
                name = '.'.join(name.split('.')[:-1]) + '.txt'
            else:
                name += '.txt'
            return name
        return False

    def do_GET(self):
        if not self.path.startswith('/Cache'):
            self.path = 'http/' + self.path
        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.requestline.startswith('POST /upload'):
            length = int(self.headers['Content-Length'])
            file_data = self.rfile.read(length)
            answer, nn_surety = self.classify(file_data)

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            result_page = self.construct_result_page(answer, nn_surety)
            self.send_header("Content-Length", str(len(result_page)))
            self.end_headers()
            self.wfile.write(result_page)
            _set_path_free(self.__cached_img_path)
        elif self.requestline.startswith('POST /translate'):
            length = int(self.headers['Content-Length'])
            to_translate = self.rfile.read(length).replace('<br>', '\r\n')

            request = urllib.unquote_plus(self.requestline)
            print request
            # from_lang_pattern = r'\?from_lang=(.*) \?'
            to_lang_pattern = r'\?to_lang=(.*) HTTP\/'
            # from_lang_full = re.findall(from_lang_pattern, request)[0]
            to_lang_full = re.findall(to_lang_pattern, request)[0]

            # from_lang = self.languages_dict[from_lang_full]
            to_lang = self.languages_dict[to_lang_full]

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            translated = translate(to_translate, to_language=to_lang)
            print to_translate
            print to_lang
            print translated
            translated = translated.encode('utf-8')
            self.send_header("Content-Length", str(len(translated)))
            self.end_headers()
            self.wfile.write(translated)

    @staticmethod
    def get_length_of_nn_data(count):
        return str([''] * count)

    def __get_dropdown_options(self):
        dropdown_options = []
        for lang in sorted(self.languages_dict):
            dropdown_options.append(
                '<a class="dropdown-item" href="#">%s</a>' % lang)
        return '\r\n'.join(dropdown_options)

    def construct_result_page(self, answer, nn_surety):
        with open('http\\result.html', 'rb') as f:
            result_page = f.read()

        image_path = self.__cached_img_path
        answer = answer.replace('\r\n', '<br>')
        dropdown_options = self.__get_dropdown_options()
        file_name = self.__get_file_name()
        avg_net_surety = numpy.mean(nn_surety)
        med_net_surety = numpy.median(nn_surety)
        nn_data_graph_len = self.get_length_of_nn_data(len(nn_surety))
        nn_data = str(nn_surety)

        result_page = result_page.format(
            image_path=image_path, answer=answer,
            dropdown_options=dropdown_options, file_name=file_name,
            avg_net_surety=avg_net_surety, med_net_surety=med_net_surety,
            length_of_nn_data=nn_data_graph_len, nn_data=str(nn_data))
        return result_page

    @Memoized
    def classify(self, file_data):
        self.__cached_img_path = _get_file_path()
        with open(self.__cached_img_path, 'wb') as img_file:
            img_file.write(file_data)

        cv2_img = cv2.imread(self.__cached_img_path, 0)
        preprocessor = Preprocessor(cv2_img)
        separated = preprocessor.separate_text()
        print 'HTTP Request Handler: Separated Chars'
        self.input_queue.put_nowait((separated, self.output_queue))
        string_text = self.wait_for_answer()
        print 'HTTP Request Handler: Classification Done!'
        auto_completed = self.spell_checker.autocomplete_text(string_text)
        print 'HTTP Request Handler: Spell Checking Done!'
        return auto_completed, [1, 2, 3]

    def wait_for_answer(self):
        while True:
            if not self.output_queue.empty():
                return self.output_queue.get_nowait()