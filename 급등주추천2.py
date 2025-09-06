# Copyright 2024 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 학습된 bert 감성분석 모델로 buy_list, buy_list, 코스피200, 코스닥150에에 있는 종목들에 대해 최근 5개의 구글링 뉴스를 추론하여 매수 여부를 결정한다.

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


def create_tbs(m1, d1, m2, d2):

    if m1 == 0:
        return 0

    start_date = datetime(2025, m1, d1)
    start_date = str(start_date)[:10]

    end_date = datetime(2025, m2, d2)
    end_date = str(end_date)[:10]

    cd_min = start_date[6:7] + '/' + start_date[8:10] + '/' + start_date[:4]
    cd_max = end_date[6:7] + '/' + end_date[8:10] + '/' + end_date[:4]

    tbs = f'cdr:1,cd_min:{cd_min},cd_max:{cd_max}'

    return tbs

def search_news_of_item(item, tbs):

    #item = '나노신소재'
    if tbs == 0:
        params = {'q': item, 'hl': 'ko', 'tbm': 'nws'}
    else:
        params = {'q' : item, 'hl' : 'ko', 'tbm' : 'nws', 'tbs' : tbs}

    header = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
    cookie = {'CONSENT' : 'YES'}
    url = 'https://www.google.com/search?'

    res = requests.get(url, params = params, headers = header, cookies = cookie)
    soup = bs(res.text, 'lxml')

    # 기사제목 파싱하는 부분
    titles = soup.find_all('div', 'n0jPhd ynAwRc MBeuO nDgy9d')
    contents = soup.find_all('div', 'GI74Re nDgy9d')
    times = soup.find_all('div', 'OSrXXb rbYSKb LfVVr')

    all_news = []
    concat_titles = ''
    n = 0
    for i in range(len(titles)):

        if times[i].get_text() not in time_pools and '분' not in times[i].get_text():
            continue

        n += 1
        if n < 5:
            concat_titles += titles[i].get_text() + ' '
            concat_titles = re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s\-\+\%\.\,\/\*\$\?\!]', ' ', concat_titles)

            news = titles[i].get_text() + " " + contents[i].get_text()
            news = re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s\-\+\%\.\,\/\*\$\?\!]', ' ', news)

            all_news.append(news)

    result = []

    result.append(item)
    result.append(concat_titles)
    result.append(all_news)

    return result

def recomm(items, offset, tbs):

    date = datetime.now().strftime('%Y-%m-%d-%H-%M')

    results = []
    for item in items:
        r = search_news_of_item(item, tbs)
        if r[2] == []:
            continue

        scores = tf.sigmoid(reloaded_model(tf.constant(np.array(r[2][:5])))).numpy()
        #scores = tf.sigmoid(reloaded_model(tf.constant(np.array([r[1]])))).numpy()
        avg_score = scores.mean()

        if offset < 0:
            if float(avg_score) < abs(offset):
                results.append([date, r[0], r[1], avg_score])
        elif float(avg_score) > offset:
            results.append([date, r[0], r[1], avg_score])

    return results

if __name__ == "__main__":

    if len(sys.argv) == 1:
        path = "buy_list.csv"
    elif sys.argv[1] == '1':
        path = "buy_list.csv"
    elif sys.argv[1] == '2':
        path = "코스피200.csv"
    elif sys.argv[1] == '3':
        path = "코스닥150.csv"

    recomm_path = '임의기간상승.csv'
    saved_model_path = './급등락_bert'
    reloaded_model = tf.saved_model.load(saved_model_path)

    time_pools = [
        "1시간 전", "2시간 전", "3시간 전", "4시간 전", "5시간 전", "6시간 전", "7시간 전", "8시간 전",
        "9시간 전", "10시간 전", "11시간 전", "12시간 전", "13시간 전", "14시간 전", "15시간 전", "16시간 전",
        "17시간 전", "18시간 전", "19시간 전", "20시간 전", "21시간 전", "22간 전", "23시간 전", "24시간 전",
        "1일 전", "2일 전", "3일 전"
    ]

    if sys.argv[1] == "all":
        keywords1 = pd.read_csv("buy_list.csv", encoding='euc-kr')['종목명'].values.tolist()
        keywords2 = pd.read_csv("코스피200.csv", encoding='euc-kr')['종목명'].values.tolist()
        keywords3 = pd.read_csv("코스닥150.csv", encoding='euc-kr')['종목명'].values.tolist()
        keywords = list(set(keywords1 + keywords2 + keywords3))
    else:
        keywords = pd.read_csv(path, encoding='euc-kr')['종목명'].values.tolist()

    # 중복 item 제거
    kyewords = list(dict.fromkeys(keywords))

    tbs = create_tbs(0, 0, 0, 0)

    if len(sys.argv) > 2:
        recomm_lists = recomm(keywords, float(sys.argv[2]), tbs)    	
    else:
        recomm_lists = recomm(keywords, 0.8, tbs)

    if recomm_lists == []:
        print('추천 종목 없음')
        exit(0)

    recomm_array = np.array(recomm_lists)

    print(recomm_array[:, 1])

    try:
        recomm_df = pd.read_csv(recomm_path, encoding='euc-kr').reset_index(drop=True)
    except:
        pd.DataFrame(recomm_array, columns=['date', 'item', 'news', 'result']).reset_index(drop=True).to_csv(recomm_path, index=False, encoding='euc-kr')
        exit(0)
    added_df = pd.DataFrame(recomm_array, columns=['date', 'item', 'news', 'result']).reset_index(drop=True)
    pd.concat([recomm_df, added_df], axis=0).to_csv(recomm_path, index=False, encoding='euc-kr')

    exit(0)

