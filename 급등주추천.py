# Copyright 2024 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 학습된 bert 감성분석 모델로 buy_list에서 있는 종목들에 대해 최근 구글링 뉴스를 추론하여 매수 여부를 결정한다.

import sys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np
import re
import requests
import lxml
from bs4 import BeautifulSoup as bs
from datetime import datetime

def search_news_of_item(item):

    #search = '유진테크'
    params = {'q' : item, 'hl' : 'ko', 'tbm' : 'nws'}#, 'tbs' : tbs}

    header = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
    cookie = {'CONSENT' : 'YES'}
    url = 'https://www.google.com/search?'

    res = requests.get(url, params = params, headers = header, cookies = cookie)
    soup = bs(res.text, 'lxml')

    # 기사제목 파싱하는 부분
    titles = soup.find_all('div', 'n0jPhd ynAwRc MBeuO nDgy9d')
    contents = soup.find_all('div', 'GI74Re nDgy9d')
    times = soup.find_all('div', 'OSrXXb rbYSKb LfVVr')
    result = []
    n = 0
    for t in titles:
        #if "시간" in times[n].get_text():
        #    n += 1
        #    continue
        news = t.get_text() + " " + contents[n].get_text().replace('"', ' ').replace("'", " ").replace('\n', ' ')

        news = re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s\-\+\%\.\,\/\*\$\?\!]', ' ', news)

        result.append(item)
        result.append(news)

        return result

    return result

def search_news_all_items(items):

    results = []
    for item in items:
        r = search_news_of_item(item)
        if r != []:
            results.append(r)
    return results

def recomm(item_news, offset):

    date = datetime.now().strftime('%Y-%m-%d')

    saved_model_path = './급등락_bert'
    reloaded_model = tf.saved_model.load(saved_model_path)

    results = tf.sigmoid(reloaded_model(tf.constant(np.array(item_news)[:, 1])))

    recomm_items = []
    for i in range(len(item_news)):
        if float(results[i][0]) > offset:
            recomm_items.append([date, item_news[i][0], item_news[i][1], results[i][0].numpy()])

    return recomm_items

if __name__ == "__main__":

    keywords = pd.read_csv('buy_list.csv', encoding='euc-kr')['종목명'].values.tolist()

    item_news = search_news_all_items(keywords)

    recomm_lists = recomm(item_news, 0.7)

    if recomm_lists == []:
        print('추천 종목 없음')
        exit(0)

    recomm_array = np.array(recomm_lists)

    print(recomm_array[:, 1])

    recomm_df = pd.read_csv('recomm_list.csv', encoding='euc-kr').reset_index(drop=True)
    added_df = pd.DataFrame(recomm_array, columns=['date', 'item', 'news', 'result']).reset_index(drop=True)
    pd.concat([recomm_df, added_df], axis=0).to_csv('recomm_list.csv', index=False, encoding='euc-kr')

    exit(0)

