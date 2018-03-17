import re
import zipfile
import urllib3
import tempfile
import shutil
from os import path
from bs4 import BeautifulSoup
import json
import numpy as np

DOWNLOAD_LINK = 'http://www.mechon-mamre.org/htmlzips/k001.zip'
POOL_MANAGER = urllib3.PoolManager()
DUMMY_SRC = r'C:\Users\bedor\Downloads\k001.zip'
INDEX_FILE = 'k0.htm'
VALID_DATA_PAGE = re.compile('k\\d{2}.*?\\.htm')

INDEX_LOOKUP = ['p', {'class': 's'}]

ILLEGAL_CHARS_REMOVER = re.compile("({.}|--|:|;|,)")


def _fetch_bible_zip():
    with POOL_MANAGER.request('GET', DOWNLOAD_LINK, preload_content=False) as resp, tempfile.TemporaryFile(
            mode='wb') as temp_fd:
        shutil.copyfileobj(resp, temp_fd)
        return temp_fd.name


def bible_files_iterator(extraction_dir):
    basedir = path.join(extraction_dir, 'k')

    def get_file(fname):
        return path.join(basedir, fname)

    pages_of_interest = []
    with open(get_file(INDEX_FILE), 'r') as index_fd:
        soup = BeautifulSoup(index_fd, 'html.parser')
        first_p = soup.find(*INDEX_LOOKUP)  # type: BeautifulSoup
        for book in first_p.find_all('a'):  # type: BeautifulSoup
            book_link = book.attrs.get('href')
            if not VALID_DATA_PAGE.match(book_link):
                continue

            pages_of_interest.append(book_link)

    for page_of_interest in pages_of_interest:
        with open(get_file(page_of_interest), 'r') as book_index:
            soup = BeautifulSoup(book_index, 'html.parser')
            everything_pointer = soup.select_one('p a:nth-of-type(2)')
            if not everything_pointer:
                print("File '", page_of_interest, "' doesn't contain an index")
                continue

            entire_book_page = everything_pointer.attrs.get('href')

        with open(get_file(entire_book_page), 'r') as entire_book_fd:
            yield entire_book_page, BeautifulSoup(entire_book_fd, 'html.parser')


def bible_book_chapters_iterator(bible_book_soup):
    """
    :type bible_book_soup: BeautifulSoup
    :rtype str
    """
    # Find actual content area
    for header in bible_book_soup.find_all('h1'):  # type: BeautifulSoup
        header_sibling = header.find_next_sibling()
        if header_sibling.name == 'p':
            book_header = header
            break

    for book_chapter_segment in book_header.find_all_next('p'):  # type: BeautifulSoup
        for some_tag in book_chapter_segment.find_all():  # type: BeautifulSoup
            if some_tag.name == 'b':
                some_tag.extract()
            else:
                some_tag.replace_with_children()

        segment_text = ILLEGAL_CHARS_REMOVER.sub(' ', book_chapter_segment.text).replace("\n", "").strip(" .").replace(
            '-', ' ')
        segment_text = re.sub(r"\s\s+", " ", segment_text)
        segment_verses = segment_text.split('.')
        if len(segment_verses[-1]) <= 1:
            segment_verses.pop()

        for segment_verse in segment_verses:
            yield segment_verse


def load_bible_data(fetch_strategy=None):
    if not fetch_strategy:
        def fetch_strategy():
            return DUMMY_SRC

    extract_path = path.join(tempfile.gettempdir(), 'bible')
    if not path.isdir(extract_path):
        src_file_name = fetch_strategy()
        with zipfile.ZipFile(src_file_name, 'r') as zf:
            zf.extractall(path=extract_path)

    bible_words = {}
    verses = []
    maximal_verse = 0
    for src_file_name, soup in bible_files_iterator(extract_path):
        for verse in bible_book_chapters_iterator(soup):
            verses.append(verse)
            verse_words = verse.split(' ')
            maximal_verse = max(maximal_verse, len(verse_words))

            for bible_word in verse_words:
                if len(bible_word) <= 1:
                    continue
                bible_words.setdefault(bible_word, 0)
                bible_words[bible_word] += 1

    dict_keypair = [(word, appearances) for word, appearances in bible_words.items()]
    dict_keypair.sort(key=lambda data: data[1])

    return verses, maximal_verse, bible_words, dict_keypair


def vectorize_bible_data(verses, maximal_verse, bible_words):
    """
    Converts dictionary to machine usable data
    :type verses: list
    :type maximal_verse: int
    :type bible_words: dict
    :rtype: list[array, dict, dict]
    """
    return None, None, None


if __name__ == '__main__':
    verses, maximal_verse, bible_words, stats = load_bible_data(None)

    print(len(bible_words.keys()))
    print('Maximal verse length: ', maximal_verse)
    print('Top 15:')
    print(stats[-20:])

    vectorized_verses, bible_words_to_index, index_to_bible_word = vectorize_bible_data(verses, maximal_verse, bible_words)
