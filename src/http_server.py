# -*- coding: utf-8 -*-
import SimpleHTTPServer
import SocketServer
import numpy
import re
import threading
import cv2
import cPickle
import urllib
import numpy as np
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
    if not path:
        return
    global used_numbers
    index_of_number = (SAVE_DIR + FILE_NAME_FORMAT).index('{')
    number_of_path = int(path[index_of_number])
    used_numbers.remove(number_of_path)


def _get_file_name_from_request(requestline):
    """
    Returns the file name from an 'upload' request. Also verifies the name is
    valid.
    :param requestline: String - the request line
    :return: String - file name
    """
    request = requestline
    pattern = r'file-name=(.*) HTTP\/'
    raw_name = re.findall(pattern, request)[0]

    # split_name = raw_name.split('%')
    # for i, char in enumerate(split_name[1:]):
    #     if all(c in string.hexdigits for c in char[:2]):
    #         char = chr(int(char[:2], 16)) + char[2:]
    #     split_name[1 + i] = char
    # name = ''.join(split_name)

    name = urllib.unquote(raw_name)
    if not re.findall(r'[\/\?\<\>\\\:\*\|\"]', name):
        if '.' in name:
            name = '.'.join(name.split('.')[:-1])
        return name
    return False


class HttpServerThread(threading.Thread):
    def __init__(self, input_queue, spell_checker):
        global used_numbers
        used_numbers = []
        threading.Thread.__init__(self)
        self.port = PORT
        self.httpd = MyTCPServer(('', self.port), MyHandler, input_queue,
                                 spell_checker)

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
        with open(TRAN_LANG_PATH, 'rb') as f:
            self.languages_dict = cPickle.load(f)
        self.result_page = None

    def finish_request(self, request, client_address):
        self.RequestHandlerClass(request, client_address,
                                 self, self.input_queue, self.spell_checker,
                                 self.languages_dict)

    def __get_dropdown_options(self):
        """
        Constructs the syntax for the language dropdown options.
        :return: A string containing all the dropdown options, ready to be
        used in the html file.
        """
        dropdown_options = []
        for lang in sorted(self.languages_dict):
            dropdown_options.append(
                '<a class="dropdown-item">%s</a>' % lang)
        return '\r\n'.join(dropdown_options)

    @staticmethod
    def __get_length_of_nn_data(count):
        """
        Returns a string representation of a list of empty strings.
        """
        return str([''] * count)

    @Memoized
    def construct_result_page(self, image_path, file_name, answer, nn_surety):
        """
        Sets up the html 'result' file. Inserts the path of the cached image,
        the answer, the language dropdown options, the name of the image, the
        average net surety, the median net surety, and the data for the NN
        surety graph.
        :param answer: String - The classified text
        :param nn_surety: A list of the surety (in percentage) of the net for
        every character classified.
        """
        with open('http\\result.html', 'rb') as f:
            result_page = f.read()

        # image_path = self.__cached_img_path
        dropdown_options = self.__get_dropdown_options()
        # file_name = _get_file_name_from_request(self.requestline)
        avg_net_surety = numpy.mean(nn_surety)
        med_net_surety = numpy.median(nn_surety)
        nn_data_graph_len = self.__get_length_of_nn_data(len(nn_surety))
        nn_data = str(nn_surety)

        result_page = result_page.format(
            image_path=image_path, answer=answer,
            dropdown_options=dropdown_options, file_name=file_name,
            avg_net_surety=avg_net_surety, med_net_surety=med_net_surety,
            length_of_nn_data=nn_data_graph_len, nn_data=str(nn_data))
        self.result_page = result_page

    def get_result_page(self):
        """
        Returns the 'result' page if it exists, and an error page if not.
        :return: The 'result' page
        """
        if self.result_page:
            return self.result_page
        with open('http\\500_error.html', 'rb') as f:
            return f.read()


class MyHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def __init__(self, request, client_address, server, input_queue,
                 spell_checker, languages_dict):
        self.server = server
        self.input_queue = input_queue
        self.output_queue = Queue()
        self.spell_checker = spell_checker
        self.__cached_img_path = ''
        self.languages_dict = languages_dict
        SimpleHTTPServer.SimpleHTTPRequestHandler.__init__(self, request,
                                                           client_address,
                                                           server)

    def do_GET(self):
        """
        Process 'GET' request. Redirects all paths to the HTTP folder (except
        for the cached images).
        """
        if not self.path.startswith('/Cache'):
            self.path = '/http' + self.path
            if self.path == '/http/results':
                self.__send_data(self.server.get_result_page())
                return
        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        """
        Process 'POST' request.
        """
        print self.requestline
        if self.requestline.startswith('POST /upload'):
            print 'HTTP Request Handler: Submit Image Request'
            self.__do_submit_image()
        elif self.requestline.startswith('POST /translate'):
            print 'HTTP Request Handler: Translate Text Request'
            self.__do_translate()

    def __send_data(self, data):
        """
        Helper function that sends the data to the client in an HTTP protocol
        :param data:
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def __do_submit_image(self):
        """
        Process an 'upload' POST request. Classifies the image, then send the
        'result' page with.
        """
        try:
            length = int(self.headers['Content-Length'])
            file_data = self.rfile.read(length)
            answer, nn_surety = self.__classify(file_data)
            image_path = self.__cached_img_path
            file_name = _get_file_name_from_request(self.requestline)

            self.server.construct_result_page(image_path, file_name, answer,
                                              nn_surety)
            self.__send_data('Success')
            # result_page = self.__construct_result_page(answer, nn_surety)
            # self.__send_data(result_page)
        except Exception, e:
            print e
            with open('http\\500_error.html', 'rb') as f:
                self.__send_data(f.read())
        _set_path_free(self.__cached_img_path)

    def __do_translate(self):
        """
        Process a 'translate' POST request.
        """
        try:
            length = int(self.headers['Content-Length'])
            to_translate = self.rfile.read(length).replace('<br>', '\r\n')

            request = urllib.unquote_plus(self.requestline)
            to_lang_pattern = r'\?to_lang=(.*) HTTP\/'
            to_lang_full = re.findall(to_lang_pattern, request)[0]
            to_lang = self.languages_dict[to_lang_full]

            translate(to_translate, to_language=to_lang)  # Warm up
            translated = translate(to_translate, to_language=to_lang)
            translated = translated.encode('utf-8')
            print 'Translated:', translated
            self.__send_data(translated)
        except Exception, e:
            print e
            error_message = 'The Server Has Experienced An Error'
            self.__send_data(error_message)
        # self.send_response(200)
        # self.send_header('Content-type', 'text/html')
        # self.send_header("Content-Length", str(len(translated)))
        # self.end_headers()
        # self.wfile.write(translated)

    # @Memoized
    # def __construct_result_page(self, answer, nn_surety):
    #     """
    #     Returns the html 'result' file. Inserts the path of the cached image,
    #     the answer, the language dropdown options, the name of the image, the
    #     average net surety, the median net surety, and the data for the NN
    #     surety graph.
    #     :param answer: String - The classified text
    #     :param nn_surety: A list of the surety (in percentage) of the net for
    #     every character classified.
    #     :return: The 'result' page
    #     """
    #     with open('http\\result.html', 'rb') as f:
    #         result_page = f.read()
    #
    #     image_path = self.__cached_img_path
    #     dropdown_options = self.__get_dropdown_options()
    #     file_name = _get_file_name_from_request(self.requestline)
    #     avg_net_surety = numpy.mean(nn_surety)
    #     med_net_surety = numpy.median(nn_surety)
    #     nn_data_graph_len = self.__get_length_of_nn_data(len(nn_surety))
    #     nn_data = str(nn_surety)
    #
    #     result_page = result_page.format(
    #         image_path=image_path, answer=answer,
    #         dropdown_options=dropdown_options, file_name=file_name,
    #         avg_net_surety=avg_net_surety, med_net_surety=med_net_surety,
    #         length_of_nn_data=nn_data_graph_len, nn_data=str(nn_data))
    #     return result_page

    @Memoized
    def __classify(self, file_data):
        """
        Classifies the text within an image.
        :param file_data: The image in byte data.
        :return: [String, List] - The answer text and the net sureties for
        every character classified.
        """
        self.__cached_img_path = _get_file_path()
        with open(self.__cached_img_path, 'wb') as img_file:
            img_file.write(file_data)

        cv2_img = cv2.imread(self.__cached_img_path, 0)
        if cv2_img is None:
            raise ServerError
        preprocessor = Preprocessor(cv2_img)
        separated = preprocessor.separate_text()
        print 'HTTP Request Handler: Separated Chars'
        self.input_queue.put_nowait((separated, self.output_queue))
        string_text = self.__wait_for_answer()
        net_sureties = self.__get_net_surety_statistics(string_text)
        print 'HTTP Request Handler: Classification Done!'
        auto_completed = self.spell_checker.autocomplete_text(string_text)
        print 'HTTP Request Handler: Spell Checking Done!'
        return auto_completed, net_sureties

    def __wait_for_answer(self):
        """
        Waits for an answer from the NN thread.
        """
        while True:
            if not self.output_queue.empty():
                result = self.output_queue.get_nowait()
                if isinstance(result, Exception):
                    raise result
                return result

    @staticmethod
    def __get_net_surety_statistics(string_text):
        """
        Returns a list of the net sureties for every character classified.
        :param string_text: The answer from the Neural Net.
        """
        sureties = []
        for line in string_text:
            for word in line:
                for char in word:
                    sureties.append([poss[1] for poss in char])
        sureties = np.array(sureties)
        return list(np.max(sureties, axis=1))


class ServerError(Exception):
    def __str__(self):
        return 'The Server Has Experienced An Error'