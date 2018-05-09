# -*- coding: utf-8 -*-
import collections
import functools


class Memoized(object):
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):  # uncacheable. a list,
        #  for instance. Better to not cache than blow up.
            return self.func(*args)
        if str(args[1:]) in self.cache:
            return self.cache[str(args[1:])]
        else:
            value = self.func(*args)
            self.cache[str(args[1:])] = value
            return value

    def __repr__(self):
        """
        Return the function's docstring.
        """
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """
        Support instance methods.
        """
        return functools.partial(self.__call__, obj)