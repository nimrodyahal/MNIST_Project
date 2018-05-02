# -*- coding: utf-8 -*-
import urllib
import HTMLParser
import re
import time
from selenium import webdriver


driver = webdriver.Chrome()

# with open('Translation Languages.txt', 'rb') as f:
#     languages_dict = cPickle.load(f)


def unescape(text):
    parser = HTMLParser.HTMLParser()
    return parser.unescape(text)


def translate(to_translate, to_language='auto', from_language='en'):
    if to_language == 'en':
        return to_translate
    link = 'https://translate.google.co.il/?hl=en#{from_lang}/' \
           '{to_lang}/{to_trans}'
    driver.get(link.format(to_lang=to_language, from_lang=from_language,
                           to_trans=urllib.quote_plus(to_translate.encode
                                                      ('utf-8'))))
    time.sleep(0.2)
    result = driver.find_element_by_id('result_box').get_attribute('innerHTML')
    final = []
    for line in result.split('<br>'):
        pattern = r'<span>(.*)</span>'
        final += re.findall(pattern, line)
    return u'\r\n'.join(final)

translate('', to_language='')