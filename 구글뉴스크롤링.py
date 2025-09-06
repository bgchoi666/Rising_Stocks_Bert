# Copyright 2024 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 일자의 상승, 하락 종목들에 대한 1일전 뉴스를 구글 뉴스에서 크롤링하여 저장한다.
# 저장 path = '급등락뉴스.csv'

import pandas as pd
import numpy as np
import re
import requests
import lxml
from bs4 import BeautifulSoup as bs
from datetime import datetime
import os

def create_tbs(y1, m1, d1, y2, m2, d2):
    start_date = datetime(y1,m1,d1)
    start_date = str(start_date)[:10]

    end_date = datetime(y2,m2,d2)
    end_date = str(end_date)[:10]

    cd_min = start_date[5:7] + '/' + start_date[8:10] + '/' + start_date[:4]
    cd_max = end_date[5:7] + '/' + end_date[8:10] + '/' + end_date[:4]

    tbs = f'cdr:1,cd_min:{cd_min},cd_max:{cd_max}'

    return tbs

time_pools = [
              "1시간 전", "2시간 전", "3시간 전", "4시간 전",# "5시간 전", "6시간 전", "7시간 전", "8시간 전",]
              #"9시간 전", "10시간 전", "11시간 전", "12시간 전", "13시간 전", "14시간 전", "15시간 전", "16시간 전",
              #"17시간 전", "18시간 전", "19시간 전", "20시간 전", "21시간 전", "22간 전", "23시간 전", "24시간 전"
             ]

def search(search, target):
    #search = '코스피'
    params = {'q' : search , 'hl' : 'ko', 'tbm' : 'nws', 'tbs' : tbs}

    header = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
    cookie = {'CONSENT' : 'YES'}
    url = 'https://www.google.com/search?'

    res = requests.get(url, params = params, headers = header, cookies = cookie)
    soup = bs(res.text, 'lxml')

    # 기사제목 파싱하는 부분
    titles = soup.find_all('div', 'n0jPhd ynAwRc MBeuO nDgy9d')
    contents = soup.find_all('div', 'GI74Re nDgy9d')
    times = soup.find_all('div', 'OSrXXb rbYSKb LfVVr')
    results = []
    date = tbs.strip("cdr:1,cd_min:")
    date = date.replace(",cd_max", "")
    result = [date]
    n = 0
    for t in titles:
        if times[n].get_text() in time_pools or "분" in times[n].get_text():
            n += 1
            continue
        # content 의 수가 title의 수 보다 작은 경우 처리
        if len(contents) < len(titles) and n == len(contents):
            break
        result.append(
            t.get_text() + " " + contents[n].get_text().replace('"', ' ').replace("'", " ").replace('\n', ' '))
        result.append(target)
        result[1] = re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s\-\+\%\.\,\/\*\$\?\!]', ' ', result[1])
        results.append(result)
        result = [date]
        n += 1

    return np.array(results)

# 검색어 입력
#keyword = input("검색어 입력: ")

if __name__ == "__main__":

    y = datetime.now().year
    m = datetime.now().month
    d = datetime.now().day

    tbs = create_tbs(2025, 9, 4, 2025, 9, 4)

    path = '하락종목/-2.8%이하_하락_2025-09-05.csv'
    if path[:4] == '상승종목':
        target = '1'
    elif path[:4] == '하락종목':
        target = '0'
    else:
        print("상승/하락 종목 파일 error")
        exit(0)

    news_path = '급등락뉴스.csv'
    if os.path.isfile(news_path):
        news_df = pd.read_csv(news_path, encoding='euc-kr').reset_index(drop=True)
    else:
        news_df = pd.DataFrame(columns=['date', 'news', 'target'])
    date = datetime.now().strftime("%Y-%m-%d")

    keyword_list = pd.read_csv(path, encoding='euc-kr')['종목명'].values.tolist()
    for keyword in keyword_list:
        results = search(keyword, target)
        if results.size == 0:
            continue
        added_news_df = pd.DataFrame(results, columns=['date', 'news', 'target']).reset_index(drop=True)
        news_df = pd.concat([news_df, added_news_df], axis=0).reset_index(drop=True)
    news_df = news_df.sample(frac=1).reset_index(drop=True)
    news_df.to_csv(news_path, index=False, encoding='euc-kr')
    exit(0)