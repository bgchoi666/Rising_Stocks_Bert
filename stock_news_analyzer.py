#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 기반 주식 뉴스 분석 통합 프로그램
Copyright 2024 Bimghi Choi. All Rights Reserved.

이 프로그램은 BERT 모델을 활용하여 한국 주식 시장(코스피, 코스닥)의 
뉴스를 분석하고 급등/급락 가능성을 예측합니다.

주요 기능:
1. 뉴스 크롤링 (구글 뉴스)
2. BERT 모델 학습
3. 종목 추천 (개별/일괄)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup as bs
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# TensorFlow 로깅 레벨 설정
tf.get_logger().setLevel('ERROR')

# ==================== 설정 및 상수 ====================
class Config:
    """프로그램 전역 설정"""
    
    # BERT 모델 설정
    BERT_MODEL_NAME = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
    TFHUB_ENCODER = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
    TFHUB_PREPROCESS = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    
    # 학습 설정
    BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 3e-5
    DROPOUT_RATE = 0.1
    TRAIN_SPLIT = 0.9
    
    # 파일 경로
    MODEL_WEIGHTS = '../model_weights.h5'
    NEWS_DATA = '../급등락뉴스.csv'
    RECOMMEND_OUTPUT = '../임의기간상승.csv'
    
    # 종목 리스트 파일
    BUY_LIST = '../buy_list.csv'
    KOSPI_200 = '../코스피200.csv'
    KOSDAQ_150 = '../코스닥150.csv'
    
    # 뉴스 검색 설정
    TIME_POOLS = [
        "1시간 전", "2시간 전", "3시간 전", "4시간 전", 
        "5시간 전", "6시간 전", "7시간 전", "8시간 전",
        "9시간 전", "10시간 전", "11시간 전", "12시간 전", 
        "13시간 전", "14시간 전", "15시간 전", "16시간 전",
        "17시간 전", "18시간 전", "19시간 전", "20시간 전", 
        "21시간 전", "22시간 전", "23시간 전", "24시간 전",
        "1일 전", "2일 전", "3일 전"
    ]
    
    # 크롤링 설정
    USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'
    GOOGLE_NEWS_URL = 'https://www.google.com/search?'


# ==================== 유틸리티 함수 ====================
class NewsUtils:
    """뉴스 크롤링 관련 유틸리티"""
    
    @staticmethod
    def create_tbs(y1, m1, d1, y2, m2, d2):
        """
        구글 검색 날짜 범위 파라미터 생성
        
        Args:
            y1, m1, d1: 시작 년, 월, 일
            y2, m2, d2: 종료 년, 월, 일
        
        Returns:
            str: tbs 파라미터 문자열
        """
        if m1 == 0:
            return 0
            
        start_date = datetime(y1, m1, d1)
        start_date_str = str(start_date)[:10]
        
        end_date = datetime(y2, m2, d2)
        end_date_str = str(end_date)[:10]
        
        cd_min = f"{start_date_str[5:7]}/{start_date_str[8:10]}/{start_date_str[:4]}"
        cd_max = f"{end_date_str[5:7]}/{end_date_str[8:10]}/{end_date_str[:4]}"
        
        return f'cdr:1,cd_min:{cd_min},cd_max:{cd_max}'
    
    @staticmethod
    def clean_text(text):
        """
        텍스트 정제 (특수문자 제거)
        
        Args:
            text: 원본 텍스트
        
        Returns:
            str: 정제된 텍스트
        """
        return re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s\-\+\%\.\,\/\*\$\?\!]', ' ', text)


class NewsCrawler:
    """구글 뉴스 크롤러"""
    
    def __init__(self):
        self.header = {'user-agent': Config.USER_AGENT}
        self.cookie = {'CONSENT': 'YES'}
    
    def search_news(self, keyword, tbs=0, target='1'):
        """
        특정 키워드에 대한 뉴스 검색
        
        Args:
            keyword: 검색 키워드 (종목명)
            tbs: 날짜 범위 파라미터
            target: '1' (상승) 또는 '0' (하락)
        
        Returns:
            numpy.array: [날짜, 뉴스내용, 타겟] 형태의 배열
        """
        # 검색 파라미터 설정
        if tbs == 0:
            params = {'q': keyword, 'hl': 'ko', 'tbm': 'nws'}
        else:
            params = {'q': keyword, 'hl': 'ko', 'tbm': 'nws', 'tbs': tbs}
        
        try:
            # 요청 및 파싱
            res = requests.get(Config.GOOGLE_NEWS_URL, params=params, 
                             headers=self.header, cookies=self.cookie, timeout=10)
            soup = bs(res.text, 'lxml')
            
            # 기사 요소 추출
            titles = soup.find_all('div', 'ilUpNd UFvD1 aSRlid IwSnJ')
            contents = soup.find_all('div', 'ilUpNd H66NU aSRlid')
            times = soup.find_all('span', 'UK5aid MDvRSc')
            
            results = []
            date = tbs.strip("cdr:1,cd_min:").replace(",cd_max", "") if tbs != 0 else datetime.now().strftime("%Y-%m-%d")
            
            # 뉴스 수집
            for i, title in enumerate(titles):
                if i >= len(times) or i >= len(contents):
                    break
                
                # 최근 뉴스만 필터링
                time_text = times[i].get_text()
                if time_text not in Config.TIME_POOLS and "분" not in time_text:
                    continue
                
                # 뉴스 내용 조합 및 정제
                news_text = f"{title.get_text()} {contents[i].get_text()}"
                news_text = NewsUtils.clean_text(news_text.replace('"', ' ').replace("'", " ").replace('\n', ' '))
                
                results.append([date, news_text, target])
            
            return np.array(results)
        
        except Exception as e:
            print(f"뉴스 검색 오류 ({keyword}): {e}")
            return np.array([])
    
    def search_for_recommendation(self, keyword, tbs=0, max_news=5):
        """
        추천을 위한 뉴스 검색 (최근 뉴스만)
        
        Args:
            keyword: 검색 키워드
            tbs: 날짜 범위
            max_news: 최대 뉴스 개수
        
        Returns:
            list: [종목명, 연결된 제목들, 개별 뉴스 리스트]
        """
        if tbs == 0:
            params = {'q': keyword, 'hl': 'ko', 'tbm': 'nws'}
        else:
            params = {'q': keyword, 'hl': 'ko', 'tbm': 'nws', 'tbs': tbs}
        
        try:
            res = requests.get(Config.GOOGLE_NEWS_URL, params=params,
                             headers=self.header, cookies=self.cookie, timeout=10)
            soup = bs(res.text, 'lxml')
            
            titles = soup.find_all('div', 'ilUpNd UFvD1 aSRlid IwSnJ')
            contents = soup.find_all('div', 'ilUpNd H66NU aSRlid')
            times = soup.find_all('span', 'UK5aid MDvRSc')
            
            all_news = []
            concat_titles = ''
            count = 0
            
            for i in range(len(titles)):
                if i >= len(times):
                    break
                
                time_text = times[i].get_text()
                if time_text not in Config.TIME_POOLS and '분' not in time_text:
                    continue
                
                count += 1
                if count <= max_news:
                    # 제목만 연결
                    concat_titles += titles[i].get_text() + ' '
                    concat_titles = NewsUtils.clean_text(concat_titles)
                    
                    # 제목 + 내용
                    news_text = f"{titles[i].get_text()} {contents[i].get_text()}"
                    news_text = NewsUtils.clean_text(news_text)
                    all_news.append(news_text)
            
            return [keyword, concat_titles, all_news]
        
        except Exception as e:
            print(f"추천용 뉴스 검색 오류 ({keyword}): {e}")
            return [keyword, '', []]


# ==================== BERT 모델 관리 ====================
class BERTModel:
    """BERT 모델 생성 및 관리"""
    
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_classifier(self):
        """
        BERT 기반 분류 모델 생성
        
        Returns:
            tf.keras.Model: 컴파일된 BERT 모델
        """
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(Config.TFHUB_PREPROCESS, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(Config.TFHUB_ENCODER, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        
        self.model = tf.keras.Model(text_input, net)
        return self.model
    
    def load_weights(self, weights_path=None):
        """
        저장된 가중치 로드
        
        Args:
            weights_path: 가중치 파일 경로
        """
        if weights_path is None:
            weights_path = Config.MODEL_WEIGHTS
        
        if os.path.isfile(weights_path):
            self.model.load_weights(weights_path)
            print(f"✓ 모델 가중치 로드 완료: {weights_path}")
        else:
            print(f"⚠ 가중치 파일 없음: {weights_path}")
    
    def save_weights(self, weights_path=None):
        """
        모델 가중치 저장
        
        Args:
            weights_path: 저장 경로
        """
        if weights_path is None:
            weights_path = Config.MODEL_WEIGHTS
        
        self.model.save_weights(weights_path)
        print(f"✓ 모델 가중치 저장 완료: {weights_path}")
    
    def compile_model(self, steps_per_epoch):
        """
        모델 컴파일 (옵티마이저, 손실함수, 메트릭 설정)
        
        Args:
            steps_per_epoch: 에폭당 스텝 수
        """
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        
        num_train_steps = steps_per_epoch * Config.EPOCHS
        num_warmup_steps = int(0.1 * num_train_steps)
        
        optimizer = optimization.create_optimizer(
            init_lr=Config.LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw'
        )
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def predict(self, texts):
        """
        뉴스 텍스트에 대한 예측
        
        Args:
            texts: 뉴스 텍스트 리스트
        
        Returns:
            numpy.array: 예측 확률값
        """
        return tf.sigmoid(self.model(tf.constant(texts))).numpy()


# ==================== 데이터 관리 ====================
class DataManager:
    """학습 데이터 관리"""
    
    @staticmethod
    def load_stock_list(file_path, encoding='euc-kr'):
        """
        종목 리스트 로드
        
        Args:
            file_path: CSV 파일 경로
            encoding: 인코딩
        
        Returns:
            list: 종목명 리스트
        """
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if '종목명' in df.columns:
                return df['종목명'].values.tolist()
            else:
                print(f"⚠ '종목명' 컬럼을 찾을 수 없음: {file_path}")
                return []
        except Exception as e:
            print(f"⚠ 파일 로드 오류 ({file_path}): {e}")
            return []
    
    @staticmethod
    def create_datasets(csv_path, train_size=60000):
        """
        학습/검증/테스트 데이터셋 생성
        
        Args:
            csv_path: 뉴스 데이터 CSV 경로
            train_size: 학습 데이터 크기
        
        Returns:
            tuple: (train_ds, val_ds, test_ds)
        """
        # 데이터 로드 및 셔플
        df = pd.read_csv(csv_path, encoding='euc-kr')
        df = df.drop(columns='date').sample(frac=1).reset_index(drop=True)
        
        # 학습 데이터 분리
        train_x = df.values[:train_size, 0].astype('str')
        train_y = df.values[:train_size, 1].astype('int32')
        
        # 학습/검증 분할
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=0.1, random_state=42
        )
        
        # TensorFlow 데이터셋 생성
        train_ds = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(train_x, dtype=tf.string),
            tf.convert_to_tensor(train_y, dtype=tf.int32)
        ))
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE).batch(Config.BATCH_SIZE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(val_x, dtype=tf.string),
            tf.convert_to_tensor(val_y, dtype=tf.int32)
        ))
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE).batch(Config.BATCH_SIZE)
        
        # 테스트 데이터
        test_x = df.values[train_size:, 0].astype('str')
        test_y = df.values[train_size:, 1].astype('int32')
        
        test_ds = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(test_x, dtype=tf.string),
            tf.convert_to_tensor(test_y, dtype=tf.int32)
        ))
        test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE).batch(Config.BATCH_SIZE)
        
        return train_ds, val_ds, test_ds


# ==================== 주요 기능 ====================
class StockNewsAnalyzer:
    """주식 뉴스 분석기 메인 클래스"""
    
    def __init__(self):
        self.crawler = NewsCrawler()
        self.model = BERTModel()
        self.data_manager = DataManager()
    
    def crawl_news(self, start_date, end_date, stock_file, rf_threshold, output_path=None):
        """
        뉴스 크롤링 및 저장
        
        Args:
            start_date: 시작 날짜 (y, m, d)
            end_date: 종료 날짜 (y, m, d)
            stock_file: 종목 리스트 파일
            rf_threshold: 등락률 임계값 (양수: 상승, 음수: 하락)
            output_path: 저장 경로
        """
        print("\n" + "="*60)
        print("뉴스 크롤링 시작")
        print("="*60)
        
        # 날짜 범위 생성
        tbs = NewsUtils.create_tbs(*start_date, *end_date)
        
        # 타겟 설정
        target = '1' if rf_threshold > 0 else '0'
        
        # 종목 리스트 로드
        keywords = self.data_manager.load_stock_list(stock_file)
        print(f"대상 종목 수: {len(keywords)}")
        
        # 기존 뉴스 데이터 로드
        if output_path is None:
            output_path = Config.NEWS_DATA
        
        if os.path.isfile(output_path):
            news_df = pd.read_csv(output_path, encoding='euc-kr')
            print(f"기존 뉴스 데이터: {len(news_df)}건")
        else:
            news_df = pd.DataFrame(columns=['date', 'news', 'target'])
            print("새로운 뉴스 데이터 생성")
        
        # 뉴스 크롤링
        total_crawled = 0
        for i, keyword in enumerate(keywords, 1):
            print(f"[{i}/{len(keywords)}] {keyword} 검색 중...", end=' ')
            results = self.crawler.search_news(keyword, tbs, target)
            
            if results.size == 0:
                print("뉴스 없음")
                continue
            
            added_df = pd.DataFrame(results, columns=['date', 'news', 'target'])
            news_df = pd.concat([news_df, added_df], axis=0).reset_index(drop=True)
            total_crawled += len(added_df)
            print(f"{len(added_df)}건 수집")
        
        # 셔플 및 저장
        news_df = news_df.sample(frac=1).reset_index(drop=True)
        news_df.to_csv(output_path, index=False, encoding='euc-kr')
        
        print(f"\n✓ 크롤링 완료: 총 {total_crawled}건 수집")
        print(f"✓ 저장 경로: {output_path}")
    
    def train_model(self, data_path=None, train_size=60000):
        """
        BERT 모델 학습
        
        Args:
            data_path: 학습 데이터 경로
            train_size: 학습 데이터 크기
        """
        print("\n" + "="*60)
        print("모델 학습 시작")
        print("="*60)
        
        if data_path is None:
            data_path = Config.NEWS_DATA
        
        # 데이터셋 생성
        print("학습 데이터 준비 중...")
        train_ds, val_ds, test_ds = self.data_manager.create_datasets(data_path, train_size)
        
        # 데이터 샘플 출력
        print("\n데이터 샘플:")
        class_names = {0: '급락', 1: '급등'}
        for text_batch, label_batch in train_ds.take(1):
            for i in range(min(3, len(text_batch))):
                print(f"  뉴스: {text_batch.numpy()[i][:100]}...")
                label = label_batch.numpy()[i]
                print(f"  라벨: {label} ({class_names[label]})\n")
        
        # 모델 생성 및 로드
        print("모델 생성 중...")
        self.model.build_classifier()
        self.model.load_weights()
        
        # 모델 컴파일
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        self.model.compile_model(steps_per_epoch)
        
        # 학습
        print(f"\n학습 시작 (에폭: {Config.EPOCHS}, 배치: {Config.BATCH_SIZE})")
        self.model.history = self.model.model.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs=Config.EPOCHS,
            verbose=1
        )
        
        # 평가
        print("\n모델 평가 중...")
        loss, accuracy = self.model.model.evaluate(test_ds)
        print(f"✓ Test Loss: {loss:.4f}")
        print(f"✓ Test Accuracy: {accuracy:.4f}")
        
        # 가중치 저장
        self.model.save_weights()
        
        # 학습 곡선 플롯
        self._plot_training_history()
    
    def _plot_training_history(self):
        """학습 이력 시각화"""
        if self.model.history is None:
            return
        
        history_dict = self.model.history.history
        
        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("✓ 학습 곡선 저장: training_history.png")
    
    def recommend_stocks(self, stock_list=None, threshold=0.8, 
                        date_range=None, output_path=None, individual_mode=False):
        """
        종목 추천
        
        Args:
            stock_list: 종목 리스트 (None이면 buy_list 사용)
            threshold: 추천 임계값 (0.0 ~ 1.0)
            date_range: 날짜 범위 (start_date, end_date)
            output_path: 결과 저장 경로
            individual_mode: 개별 종목 모드
        """
        print("\n" + "="*60)
        print("종목 추천 시작")
        print("="*60)
        
        # 모델 로드
        print("모델 로드 중...")
        self.model.build_classifier()
        self.model.load_weights()
        
        # 종목 리스트 준비
        if stock_list is None:
            stock_list = self.data_manager.load_stock_list(Config.BUY_LIST)
        
        # 중복 제거
        stock_list = list(dict.fromkeys(stock_list))
        print(f"분석 대상 종목: {len(stock_list)}개")
        
        # 날짜 범위 설정
        if date_range is None:
            tbs = 0  # 전체 기간
        else:
            tbs = NewsUtils.create_tbs(*date_range[0], *date_range[1])
        
        # 추천 로직
        results = []
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        
        for i, stock in enumerate(stock_list, 1):
            print(f"[{i}/{len(stock_list)}] {stock} 분석 중...", end=' ')
            
            # 뉴스 검색
            news_data = self.crawler.search_for_recommendation(stock, tbs, max_news=5)
            
            if not news_data[2]:  # 뉴스가 없으면
                print("뉴스 없음")
                continue
            
            # 예측
            scores = self.model.predict(np.array(news_data[2][:5]))
            avg_score = scores.mean()
            
            # 임계값 확인
            if individual_mode or (threshold >= 0 and avg_score > threshold) or \
               (threshold < 0 and avg_score < abs(threshold)):
                results.append([current_time, news_data[0], news_data[1], avg_score])
                print(f"점수: {avg_score:.4f} ✓")
            else:
                print(f"점수: {avg_score:.4f}")
        
        # 결과 처리
        if not results:
            print("\n추천 종목 없음")
            return
        
        # DataFrame 생성 및 정렬
        result_df = pd.DataFrame(results, columns=['date', 'item', 'news', 'result'])
        result_df = result_df.sort_values(by='result', ascending=False).reset_index(drop=True)
        
        # 저장
        if output_path is None:
            output_path = Config.RECOMMEND_OUTPUT
        
        try:
            existing_df = pd.read_csv(output_path, encoding='euc-kr')
            result_df = pd.concat([existing_df, result_df], axis=0)
        except:
            pass
        
        result_df.to_csv(output_path, index=False, encoding='euc-kr')
        
        # 결과 출력
        print("\n" + "="*60)
        print(f"추천 종목: {len(results)}개")
        print("="*60)
        for idx, row in result_df.head(10).iterrows():
            print(f"{idx+1}. {row['item']} - 점수: {row['result']:.4f}")
        
        print(f"\n✓ 결과 저장: {output_path}")


# ==================== CLI 인터페이스 ====================
def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='BERT 기반 주식 뉴스 분석 통합 프로그램',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 뉴스 크롤링
  python stock_news_analyzer.py crawl --start 2026 1 15 --end 2026 1 15 --file buy_list.csv --threshold 10
  
  # 모델 학습
  python stock_news_analyzer.py train --data 급등락뉴스.csv --train-size 60000
  
  # 종목 추천 (일괄)
  python stock_news_analyzer.py recommend --threshold 0.8 --list all
  
  # 종목 추천 (개별)
  python stock_news_analyzer.py recommend --stocks 삼성전자 SK하이닉스
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='실행할 작업')
    
    # 크롤링 명령
    crawl_parser = subparsers.add_parser('crawl', help='뉴스 크롤링')
    crawl_parser.add_argument('--start', nargs=3, type=int, required=True,
                             metavar=('Y', 'M', 'D'), help='시작 날짜 (년 월 일)')
    crawl_parser.add_argument('--end', nargs=3, type=int, required=True,
                             metavar=('Y', 'M', 'D'), help='종료 날짜 (년 월 일)')
    crawl_parser.add_argument('--file', required=True, help='종목 리스트 파일')
    crawl_parser.add_argument('--threshold', type=float, required=True,
                             help='등락률 임계값 (양수: 상승, 음수: 하락)')
    crawl_parser.add_argument('--output', default=Config.NEWS_DATA,
                             help='출력 파일 경로')
    
    # 학습 명령
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--data', default=Config.NEWS_DATA,
                             help='학습 데이터 경로')
    train_parser.add_argument('--train-size', type=int, default=60000,
                             help='학습 데이터 크기')
    
    # 추천 명령
    recommend_parser = subparsers.add_parser('recommend', help='종목 추천')
    recommend_parser.add_argument('--list', choices=['buy', 'kospi', 'kosdaq', 'all'],
                                 help='종목 리스트 선택')
    recommend_parser.add_argument('--stocks', nargs='+', help='개별 종목명')
    recommend_parser.add_argument('--threshold', type=float, default=0.8,
                                 help='추천 임계값')
    recommend_parser.add_argument('--output', default=Config.RECOMMEND_OUTPUT,
                                 help='결과 저장 경로')
    
    args = parser.parse_args()
    
    # 명령어 처리
    analyzer = StockNewsAnalyzer()
    
    if args.command == 'crawl':
        analyzer.crawl_news(
            start_date=tuple(args.start),
            end_date=tuple(args.end),
            stock_file=args.file,
            rf_threshold=args.threshold,
            output_path=args.output
        )
    
    elif args.command == 'train':
        analyzer.train_model(
            data_path=args.data,
            train_size=args.train_size
        )
    
    elif args.command == 'recommend':
        # 종목 리스트 결정
        if args.stocks:
            stock_list = args.stocks
        elif args.list:
            if args.list == 'buy':
                stock_list = analyzer.data_manager.load_stock_list(Config.BUY_LIST)
            elif args.list == 'kospi':
                stock_list = analyzer.data_manager.load_stock_list(Config.KOSPI_200)
            elif args.list == 'kosdaq':
                stock_list = analyzer.data_manager.load_stock_list(Config.KOSDAQ_150)
            elif args.list == 'all':
                stock_list = (
                    analyzer.data_manager.load_stock_list(Config.BUY_LIST) +
                    analyzer.data_manager.load_stock_list(Config.KOSPI_200) +
                    analyzer.data_manager.load_stock_list(Config.KOSDAQ_150)
                )
        else:
            stock_list = None
        
        analyzer.recommend_stocks(
            stock_list=stock_list,
            threshold=args.threshold,
            output_path=args.output,
            individual_mode=bool(args.stocks)
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
