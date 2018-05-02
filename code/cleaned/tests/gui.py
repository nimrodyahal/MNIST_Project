# -*- coding: utf-8 -*-
import wx
import socket
import cPickle as Pickle
from docx import Document


ADDRESS = 'mnistproject.ddns.net'
SIZE = (600, 350)
# Load Image Button
LOAD_BUTTON_SIZE = (80, 35)
LOAD_BUTTON_POS = (450, 100)
# Classify Button
CLASSIFY_BUTTON_SIZE = (80, 35)
CLASSIFY_BUTTON_POS = (450, 150)
# File Dialog
DIALOG_TITLE = 'Open'
FILE_EXTENSION_NAME = 'Image Files'
FILE_EXTENSIONS = '*.png;*.jpg;*jpeg;*.gif;*.bmp;*.png'
# Bitmap
BITMAP_HEIGHT = 250
MAX_BITMAP_WIDTH = 390
BITMAP_POS = (30, 30)


class ModdedFrame(wx.Frame):
    def __init__(self, parent, _title, client_socket):
        wx.Frame.__init__(self, parent, title=_title, size=SIZE)
        self.panel = wx.Panel(self)
        self.client_socket = client_socket

        self.load_image_button = self.init_load_button()
        self.classify_button = self.init_classify_button()
        self.image, self.image_path, self.open_bitmap = self.init_image()
        self.status_bar = self.CreateStatusBar(1)

        self.Center()

    def init_load_button(self):
        """
        Initialise the 'load image' button.
        :return:
        """
        load_image_button = wx.Button(self.panel, label='Load Image',
                                      size=LOAD_BUTTON_SIZE,
                                      pos=LOAD_BUTTON_POS)
        load_image_button.Bind(wx.EVT_BUTTON, self.on_load_press)
        return load_image_button

    def init_classify_button(self):
        """
        Initialise the 'classify' button.
        :return:
        """
        classify_button = wx.Button(self.panel, label='Classify',
                                    size=CLASSIFY_BUTTON_SIZE,
                                    pos=CLASSIFY_BUTTON_POS)
        classify_button.Bind(wx.EVT_BUTTON, self.on_classify_press)
        return classify_button

    def init_image(self):
        """
        Initialize the image.
        :return: [the image, the image path, the file dialog]
        """
        image = image_path = None
        open_bitmap = wx.FileDialog(self, DIALOG_TITLE, '', '',
                                    FILE_EXTENSION_NAME + '|' +
                                    FILE_EXTENSIONS,
                                    wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        return image, image_path, open_bitmap

    def on_load_press(self, event):
        """
        When pressing the 'Load Image' button, open a file dialog prompt, then
        align it to fit in the frame.
        """
        #  Undisplay previous image
        if type(self.image) == wx.StaticBitmap:
            self.image.Hide()
        # Choose image
        self.open_bitmap.ShowModal()
        self.image_path = self.open_bitmap.GetPath()
        if not self.image_path:
            return
        bitmap = wx.Bitmap(self.image_path)
        # Rescale image
        bitmap_size = bitmap.GetWidth(), bitmap.GetHeight()
        scaling = self.get_resize_scale(bitmap_size)
        width, height = scaling * bitmap_size[0], scaling * bitmap_size[1]
        # Display image
        image = bitmap.ConvertToImage().Scale(width, height,
                                              wx.IMAGE_QUALITY_HIGH)
        bitmap = wx.Bitmap(image)
        self.image = wx.StaticBitmap(self, -1, bitmap, pos=BITMAP_POS)

    def on_classify_press(self, event):
        """
        When pressing the 'Classify' button, send the server the image to
        process if it was chosen. If not, alert the user to choose an image.
        Open a message box containing the classified text, and prompt the user
        to save it to a file.
        """
        # Display error message on the status bar if no image was loaded
        if type(self.image) != wx.StaticBitmap:
            self.status_bar.SetStatusText('You must first load an image to'
                                          ' classify')
            return
        # Clear status bar
        self.status_bar.SetStatusText('')
        # Send for classification
        with open(self.image_path, 'rb') as img:
            self.client_socket.send(img.read())

        response = str(Pickle.loads(self.client_socket.recv(819200)))

        dialog = wx.MessageDialog(self, response + '\r\nWould you like to '
                                                   'save this text to a file?',
                                  'Text',
                                  wx.YES_NO)
        if dialog.ShowModal() == wx.ID_YES:
            self.save_text_file(response)

    def save_text_file(self, text):
        """
        Prompts the user to save the text to a file.
        :param text: The text to save to the file.
        """
        with wx.FileDialog(
                self, "Save Text File",
                wildcard="Word Files (*.docx)|*.docx|Text Files (*.txt)|*.txt",
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            # the user changed their mind
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            # save the current contents in the file
            path = fileDialog.GetPath()
            try:
                self.do_save_data(path, text)
            except IOError:
                wx.LogError("Cannot save current data in file '%s'." % path)

    @staticmethod
    def do_save_data(path, text):
        """
        Saves the data to a file. Either Word Document or .txt file.
        :param path: The path of the file.
        :param text: The text to save to the file.
        """
        if path.endswith('.docx'):
            document = Document()
            print repr(text)
            document.add_paragraph(str(text))
            document.save(path)
        elif path.endswith('.txt'):
            with open(path, 'w') as f:
                f.write(text)

    @staticmethod
    def get_resize_scale(bitmap_size):
        """
        Returns the scaling required to get a bitmap to fit to it's specified
        height.
        :param bitmap_size: The size of the bitmap (width, height)
        """
        scaling = BITMAP_HEIGHT / (bitmap_size[1] * 1.0)
        rescaled_width = bitmap_size[0] * scaling
        if rescaled_width > MAX_BITMAP_WIDTH:
            scaling = MAX_BITMAP_WIDTH / (bitmap_size[0] * 1.0)
        return scaling


def main():
    client_socket = socket.socket()
    client_socket.connect((ADDRESS, 500))

    app = wx.App(False)
    frame = ModdedFrame(None, 'Optical Character Recognition', client_socket)
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()