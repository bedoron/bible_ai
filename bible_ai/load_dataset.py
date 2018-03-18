import re
import zipfile
import urllib3
import tempfile
import shutil
from os import path, remove
from bs4 import BeautifulSoup
import pickle
from pathlib import Path
import numpy as np

DOWNLOAD_LINK = 'http://www.mechon-mamre.org/htmlzips/k001.zip'


class BibleLoader(object):
    TEMP_FILE_DOWNLOAD_NAME = 'bible_temp'
    POOL_MANAGER = urllib3.PoolManager()
    INDEX_FILE = 'k0.htm'
    INDEX_LOOKUP = ['p', {'class': 's'}]
    VALID_DATA_PAGE = re.compile('k\\d{2}.*?\\.htm')
    ILLEGAL_CHARS_REMOVER = re.compile("({.}|--|:|;|,)")
    CACHE = 'bible_cache'

    def load(self, force_fetch=False):
        # Check if cache exists - if so - load from it
        # if not - start loading from internets and cache it
        if force_fetch:
            remove(self._resources_cache)

        cache_path = Path(self._resources_cache)
        if cache_path.is_file():
            print("Loading from cache: ", self._resources_cache)
            with cache_path.open('rb') as f:
                self.__dict__.update(pickle.load(f).__dict__)
        else:
            self._load_from_web()
            self._store_to_cache()

    def _store_to_cache(self):
        with open(self._resources_cache, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def __getstate__(self):
        return {'verses': self.verses, 'maximal_verse': self.maximal_verse, 'bible_words': self.bible_words,
                'dict_keypair': self.dict_keypair}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _load_from_web(self):
        download_dst = BibleLoader._fetch_bible_from_url()
        with zipfile.ZipFile(download_dst, 'r') as zf:
            zf.extractall(path=self._extract_zip_to)

        self.verses, self.maximal_verse, self.bible_words, self.dict_keypair = self._load_bible_data()

    def __init__(self, download_link) -> None:
        super().__init__()
        self._download_link = download_link
        self._resources_cache = path.join(tempfile.gettempdir(), BibleLoader.CACHE)
        self._extract_zip_to = path.join(tempfile.gettempdir(), 'bible')

    @staticmethod
    def _fetch_bible_from_url():
        download_dst = path.join(tempfile.gettempdir(), BibleLoader.TEMP_FILE_DOWNLOAD_NAME)
        with BibleLoader.POOL_MANAGER.request('GET', DOWNLOAD_LINK, preload_content=False) as resp, open(
                download_dst, mode='wb') as temp_fd:
            shutil.copyfileobj(resp, temp_fd)
            return download_dst

    @staticmethod
    def _bible_files_iterator(extracted_bible_files_location):
        basedir = path.join(extracted_bible_files_location, 'k')

        def get_file(fname):
            return path.join(basedir, fname)

        pages_of_interest = []
        with open(get_file(BibleLoader.INDEX_FILE), 'r') as index_fd:
            soup = BeautifulSoup(index_fd, 'html.parser')
            first_p = soup.find(*BibleLoader.INDEX_LOOKUP)  # type: BeautifulSoup
            for book in first_p.find_all('a'):  # type: BeautifulSoup
                book_link = book.attrs.get('href')
                if not BibleLoader.VALID_DATA_PAGE.match(book_link):
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

    @staticmethod
    def _bible_book_chapters_iterator(bible_book_soup):
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

            segment_text = BibleLoader.ILLEGAL_CHARS_REMOVER.sub(' ', book_chapter_segment.text).replace("\n",
                                                                                                         "").strip(
                " .").replace(
                '-', ' ')
            segment_text = re.sub(r"\s\s+", " ", segment_text)
            segment_verses = segment_text.split('.')
            if len(segment_verses[-1]) <= 1:
                segment_verses.pop()

            for segment_verse in segment_verses:
                yield segment_verse

    def fetch_bible_data(self, fetch_strategy=_fetch_bible_from_url, destination=None):
        if not fetch_strategy:
            def fetch_strategy():
                return DUMMY_SRC

        if path.isdir(destination):
            shutil.rmtree(destination)

        downloaded_zip_location = fetch_strategy()
        with zipfile.ZipFile(downloaded_zip_location, 'r') as zf:
            zf.extractall(path=destination)
            return destination

    def _load_bible_data(self):
        bible_words = {}
        verses = []
        maximal_verse = 0
        for src_file_name, soup in self._bible_files_iterator(self._extract_zip_to):
            for verse in self._bible_book_chapters_iterator(soup):
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


if __name__ == '__main__':
    bl = BibleLoader(DOWNLOAD_LINK)
    bl.load()

    print(len(bl.bible_words.keys()))
    print('Maximal verse length: ', bl.maximal_verse)
    print('Total verses: ', len(bl.verses))
    print('Top 15:')

    for stat in bl.dict_keypair[-20:]:
        print(stat)
