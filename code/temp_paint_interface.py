from PIL import Image


with open('test_byte_stuff', 'wb') as f:
    im = Image.open("test_draw.png")
    pix = im.load()
    print im.size
    for y in xrange(28):
        for x in xrange(28):
            if pix[x, y] == (0, 0, 0):
                f.write(chr(0))
            else:
                f.write(chr(9))
