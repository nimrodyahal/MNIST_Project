# -*- coding: utf-8 -*-
import wx
from neural_network import load_data_shared
from pre_processing import load_img_arr
from handle_nn import load_net, load_multi_net, train_multi_net


SIZE = (600, 350)
# Load Image Button
LOAD_BUTTON_SIZE = (80, 35)
LOAD_BUTTON_POS = (450, 100)
# Classify Button
CLASSIFY_BUTTON_SIZE = (80, 35)
CLASSIFY_BUTTON_POS = (450, 150)
# Bitmap
BITMAP_HEIGHT = 250
MAX_BITMAP_WIDTH = 390
BITMAP_POS = (30, 30)

mapping = load_data_shared()[2]


class ModdedFrame(wx.Frame):
    def __init__(self, parent, _title, multi_net):
        wx.Frame.__init__(self, parent, title=_title, size=SIZE)
        self.panel = wx.Panel(self)
        self.multi_net = multi_net

        # Initialise The Refresh Button
        self.load_image_button = wx.Button(self.panel, label='Load Image',
                                           size=LOAD_BUTTON_SIZE,
                                           pos=LOAD_BUTTON_POS)
        self.load_image_button.Bind(wx.EVT_BUTTON, self.on_load_press)

        # Initialise The Refresh Button
        self.classify_button = wx.Button(self.panel, label='Classify',
                                         size=CLASSIFY_BUTTON_SIZE,
                                         pos=CLASSIFY_BUTTON_POS)
        self.classify_button.Bind(wx.EVT_BUTTON, self.on_classify_press)

        # Initialize The Image
        self.image = self.image_path = None
        self.open_bitmap = wx.FileDialog(self, "Open", "", "",
                                         "Image files (*.png)|*.png",
                                         wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        # Initialize The Status Bar
        self.status_bar = self.CreateStatusBar(1)

        self.Center()

    def on_load_press(self, event):
        #  Undisplay previous image
        if type(self.image) == wx.StaticBitmap:
            self.image.Hide()
        # Choose image
        self.open_bitmap.ShowModal()
        self.image_path = self.open_bitmap.GetPath()
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
        # Display error message on the status bar if no image was loaded
        if type(self.image) != wx.StaticBitmap:
            self.status_bar.SetStatusText('You must first load an image to'
                                          ' classify')
            return
        # Clear status bar
        self.status_bar.SetStatusText('')
        # Classify
        char = load_img_arr(self.image_path)
        # classification = self.multi_net.feedforward(char)[0].argsort()[::-1][:3]
        # print [chr(mapping[c]) for c in classification]
        classifications = self.multi_net.feedforward(char)
        print classifications

    @staticmethod
    def get_resize_scale(bitmap_size):
        scaling = BITMAP_HEIGHT / (bitmap_size[1] * 1.0)
        rescaled_width = bitmap_size[0] * scaling
        if rescaled_width > MAX_BITMAP_WIDTH:
            scaling = MAX_BITMAP_WIDTH / (bitmap_size[0] * 1.0)
        return scaling


def main():
    multi_net = train_multi_net(1)
    # multi_net = load_multi_net(['..\\Saved Nets\\test_net0.txt'])

    app = wx.App(False)
    frame = ModdedFrame(None, 'Process Privileges', multi_net)
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()