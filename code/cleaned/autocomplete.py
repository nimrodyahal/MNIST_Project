import re
from collections import Counter
import string
import numpy as np
from copy import deepcopy


def get_many_words(text):
    """
    Returns a list of all words in the text.
    :param text: The text in question.
    """
    return re.findall(r'\w+', text.lower())


def unwrap_to_chars(func):
    """
    Decorator. Makes the function iterate over every character in every word in
    every line in the text.
    """
    def decorator(text, *args):
        for l_i, line in enumerate(text):
            for w_i, word in enumerate(line):
                for c_i, char in enumerate(word):
                    word[c_i] = func(char, *args)
                line[w_i] = word
            text[l_i] = line
        return text
    return decorator


class SpellChecker():
    def __init__(self, database_path):
        with open(database_path, 'r') as f:
            self.word_counter = Counter(get_many_words(f.read()))

    def prob_of_word(self, word):
        """
        Returns probability of `word`.
        :param word: The word in question
        """
        n = sum(self.word_counter.values())
        return self.word_counter[word] / n

    def known(self, words):
        """
        Returns the subset of `words` that appear in the dictionary of WORDS.
        :param words: The words in question
        """
        return set(w for w in words if w in self.word_counter)

    def candidates(self, word):
        """
        Returns a possible spelling corrections for word.
        :param word: The word in question
        """
        return self.known([word]) or self.known(self.__edits1(word)) or \
            self.known(self.__edits2(word)) or [word]

    def correction(self, word):
        """
        Returns the most probable spelling correction for word.
        :param word: The word in question
        """
        if len(word) == 0:
            return word
        word = word.lower()
        return max(self.candidates(word), key=self.prob_of_word)

    @staticmethod
    def __edits1(word):
        """
        Returns all edits that are one edit away from `word`.
        :param word: The word in question
        """
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [l + r[1:] for l, r in splits if r]
        transposes = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1]
        replaces = [l + c + r[1:] for l, r in splits if r for c in letters]
        inserts = [l + c + r for l, r in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def __edits2(self, word):
        """
        Returns all edits that are two edits away from `word`.
        :param word: The word in question
        """
        return (e2 for e1 in self.__edits1(word) for e2 in self.__edits1(e1))

    @staticmethod
    @unwrap_to_chars
    def __recalculate_digits(char):
        """
        Changes the net's surety about digits to be half of what they are.
        That is because digits are much less likely to crop up in a text.
        NOTE: this function is decorated to iterate over all chars in the
        texts, so while this looks like it only works on one char, it actually
        works on all the text.
        :param char: The char in question (Read NOTE).
        """
        for t_i, trial in enumerate(char):
            if trial[0].isdigit():
                trial = (trial[0], trial[1] / 2)
            char[t_i] = trial
        return sorted(char, key=lambda x: x[1], reverse=True)

    @staticmethod
    @unwrap_to_chars
    def __remove_duplicates(char):
        """
        Removes duplicate letters in the form of lower\capital letters (e.g
        i and I, n and N, etc.). This is because the letters are being lower
        cased eventually anyway.
        NOTE: this function is decorated to iterate over all chars in the
        texts, so while this looks like it only works on one char, it actually
        works on all the text.
        :param char: The char in question (Read NOTE).
        """
        possibilities = ''
        real_char = []
        for t_i, trial in enumerate(char):
            if trial[0].lower() not in possibilities:
                possibilities += trial[0].lower()
                real_char.append(trial)
        return real_char

    @staticmethod
    def __attempt_generator(word, tries_per_char):
        chars = [c[0][0] for c in word]
        yield ''.join(chars).lower()
        for i in range(tries_per_char * len(word)):
            chances = []
            for char in word:
                if len(char) > 1:
                    chances.append(abs(char[0][1] - char[1][1]) / char[0][1])
                else:
                    chances.append(1)
            if chances == [1] * len(chances):
                return
            word[np.argmin(chances)].pop(0)
            chars = [c[0][0] for c in word]
            yield ''.join(chars).lower()

    def autocomplete_text(self, text):
        text = self.__recalculate_digits(text)
        text = self.__remove_duplicates(text)
        final_text = []
        for line in text:
            final_line = []
            for word in line:
                final_word = ''
                attempts = self.__attempt_generator(deepcopy(word), 3)
                for attempt in attempts:
                    if self.known([attempt]):
                        final_word = attempt
                        break
                if not final_word:
                    attempts = self.__attempt_generator(word, 3)
                    first_attempt = attempts.next()
                    final_word = self.correction(first_attempt)
                    if final_word == first_attempt:
                        second_attempt = attempts.next()
                        final_word = self.correction(second_attempt)
                        if final_word == second_attempt:
                            final_word = first_attempt
                final_line.append(final_word)
            final_text.append(' '.join(final_line))
        return '\r\n'.join(final_text)
