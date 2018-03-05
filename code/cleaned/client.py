import socket


def main():
    client_socket = socket.socket()
    client_socket.connect(('127.0.0.1', 500))
    image_path = 'C:\Users\dev\Desktop\\nimrodyahal\MNIST_Project-master\code\\test_draw.png'
    with open(image_path, 'rb') as img:
        client_socket.send(img)


if __name__ == '__main__':
    main()