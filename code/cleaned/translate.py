# -*- coding: utf-8 -*-
import urllib
import re
import time
from selenium import webdriver


driver = webdriver.Chrome()


def translate(to_translate, to_language='auto', from_language='en'):
    """
    Translates a text to another language. Uses Google Translate.
    to_language and from_lanuage have to be one of the language codes Google
    recognizes. For a full list of language codes, visit the URL:
    https://cloud.google.com/translate/docs/languages
    :param to_translate: The text to translate
    :param to_language: The language to translate the text to.
    :param from_language: The original language of the text.
    Default: 'en'(English)
    :return: The translated text
    """
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

translate('', to_language='')  # To "warm up"