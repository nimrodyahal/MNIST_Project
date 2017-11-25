# -*- coding: utf-8 -*-


def main():
    num_layers = 10
    for l in xrange(2, num_layers):
        print -l
        print 'delta', -l + 1
        print 'nabla_w', - l - 1

    print '\r\n'

    for l in xrange(-2, -num_layers, -1):
        print l
        print 'delta', l + 1
        print 'nabla_w', l - 1


if __name__ == '__main__':
    main()