#! /usr/bin/env python

import urllib2
import os
import json
from bing_search_api import BingSearchAPI
import string
import sys

def read_bing_key():
    with open('bing_key.txt', 'r') as f:
        bing_key = f.read()
    bing_key = bing_key.replace('\n','')
    return bing_key

def get_format(format):
    format_list = format.split("?")
    if (len(format_list) > 1):
        format = format_list[0]

    format_list = format.split("%")
    if (len(format_list) > 1):
        format = format_list[0]
    return format

def download_single_image(url, search_query, title_name):
    format = url.split('.')[-1]
    format = get_format(format)

    if (format == 'gif'):
        print('gif')
        return

    dir_name = "image/" + search_query.replace(' ','_')
    if not (os.path.isdir(dir_name)):
        os.mkdir(dir_name)

    valid_chars = "-_() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in title_name if c in valid_chars)
    req = urllib2.Request(url)
    req.add_header('Accept', 
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')
    req.add_header('user-agent', 
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_4) AppleWebKit/537.13 (KHTML, like Gecko) Chrome/24.0.1290.1 Safari/537.13')
    try:
        webpage = urllib2.urlopen(req).read()
        full_file_name = dir_name + '/' + filename + '.' + format
        f = open(full_file_name, 'wb')
        f.write(webpage)
        f.close()
    except:
        print(url)
        
def crawl_from_bing(search_query):
    my_key = read_bing_key()
    # search_query = "nba jumpshot"
    bing = BingSearchAPI(my_key)
    for i in range(20):
        params = {
              '$format': 'json',
              '$top': 50,
              '$skip': i * 50}
        result_list = bing.search('image',search_query,params).json()
        print(len(result_list['d']['results'][0]['Image']))
        for result in result_list['d']['results'][0]['Image']:
            image_url = (result['MediaUrl'])
            title_name = result['Title'].encode('gbk', 'ignore').decode(encoding="utf-8", errors="ignore")
            title_name = title_name.replace('... ','')
            download_single_image(image_url, search_query, title_name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: crawler search_query")
        exit(1)
    search_query = ""
    for i in range(len(sys.argv)):
        if (i == 0):
            continue
        search_query += sys.argv[i] + " "

    search_query = search_query[:-1]
    crawl_from_bing(search_query)
