import re
from collections import Counter
import string
from time import time
import numpy as np
from copy import deepcopy


def get_many_words(text):
    return re.findall(r'\w+', text.lower())


WORDS = Counter(get_many_words(open('big_merged.txt').read()))


def p(word, n=sum(WORDS.values())):
    """Probability of `word`."""
    return WORDS[word] / n


def correction(word):
    """Most probable spelling correction for word."""
    if len(word) == 0:
        return word
    word = word.lower()
    return max(candidates(word), key=p)


def candidates(word):
    """Generate possible spelling corrections for word."""
    return known([word]) or known(edits1(word)) or known(edits2(word))\
        or [word]


def known(words):
    """The subset of `words` that appear in the dictionary of WORDS."""
    return set(w for w in words if w in WORDS)


def edits1(word):
    """All edits that are one edit away from `word`."""
    letters = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def recalculate_digit(text):
    for l_i, line in enumerate(text):
        for w_i, word in enumerate(line):
            for c_i, char in enumerate(word):
                for t_i, trial in enumerate(char):
                    if trial[0].isdigit():
                        trial = (trial[0], trial[1] / 2)
                    char[t_i] = trial
                char = sorted(char, key=lambda x: x[1], reverse=True)
                word[c_i] = char
            line[w_i] = word
        text[l_i] = line
    return text


def remove_duplicates(text):
    for l_i, line in enumerate(text):
        for w_i, word in enumerate(line):
            for c_i, char in enumerate(word):
                possibilities = ''
                real_char = []
                for t_i, trial in enumerate(char):
                    if trial[0].lower() not in possibilities:
                        possibilities += trial[0].lower()
                        real_char.append(trial)
                word[c_i] = real_char
            line[w_i] = word
        text[l_i] = line
    return text


def attempt_generator(word, tries_per_char, logging=False):
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
        if logging:
            print 'chances:', chances
            print 'word:', word
        chars = [c[0][0] for c in word]
        yield ''.join(chars).lower()


def main():
    time0 = time()
    with open('bla.txt', 'r') as f:
        text = eval(f.read())

    text = recalculate_digit(text)
    text = remove_duplicates(text)
    final_text = []
    for line in text:
        final_line = []
        for word in line:
            final_word = ''
            attempts = attempt_generator(deepcopy(word), 3)
            for attempt in attempts:
                if known([attempt]):
                    final_word = attempt
                    break
            if not final_word:
                attempts = attempt_generator(word, 3)
                first_attempt = attempts.next()
                final_word = correction(first_attempt)
                if final_word == first_attempt:
                    second_attempt = attempts.next()
                    final_word = correction(second_attempt)
                    if final_word == second_attempt:
                        final_word = first_attempt
            final_line.append(final_word)
        final_text.append(' '.join(final_line))
    print '\r\n'.join(final_text)
    print time() - time0

if __name__ == '__main__':
    main()