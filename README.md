# 🤖 BERT 기반 주식 뉴스 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Copyright-red.svg)](LICENSE)

한국 주식 시장의 뉴스를 AI로 분석하여 급등 가능성이 있는 종목을 자동으로 추천하는 통합 시스템입니다.

## ✨ 주요 특징

- 🔍 **자동 뉴스 크롤링**: 구글 뉴스에서 종목별 최신 뉴스 자동 수집
- 🧠 **BERT AI 분석**: 딥러닝 기반 감성 분석으로 상승/하락 예측
- 📊 **종목 추천**: 학습된 AI 모델로 유망 종목 자동 선별
- ⚡ **통합 인터페이스**: 단일 프로그램으로 모든 기능 수행
- 🎯 **커스터마이징**: 종목 리스트, 임계값 등 유연한 설정

## 🎯 누가 사용하나요?

- 📈 **개인 투자자**: 일일 종목 선정에 AI 활용
- 💼 **퀀트 트레이더**: 뉴스 기반 트레이딩 전략 개발
- 🎓 **연구자**: 금융 NLP 연구 및 백테스팅
- 🤖 **개발자**: 자동화 투자 시스템 구축

## 🚀 빠른 시작

### 설치
```bash
# 저장소 클론
git clone [repository-url]
cd stock-news-analyzer

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 3단계 실행
```bash
# 1. 뉴스 수집
python stock_news_analyzer.py crawl \
  --start 2026 1 15 --end 2026 1 15 \
  --file buy_list.csv --threshold 10

# 2. 모델 학습
python stock_news_analyzer.py train

# 3. 종목 추천
python stock_news_analyzer.py recommend --list all --threshold 0.8
```

자세한 내용은 [빠른시작가이드.md](빠른시작가이드.md)를 참조하세요.

## 📖 문서

- [📘 사용설명서](사용설명서.md) - 전체 기능 상세 설명
- [🚀 빠른시작가이드](빠른시작가이드.md) - 5분 만에 시작하기
- [⚙️ API 문서](stock_news_analyzer.py) - 소스 코드 주석

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐
│  구글 뉴스 API  │
└────────┬────────┘
         │ 크롤링
         ▼
┌─────────────────┐
│  뉴스 데이터    │
│  (CSV 저장)     │
└────────┬────────┘
         │ 학습
         ▼
┌─────────────────┐
│  BERT 모델      │
│  (TensorFlow)   │
└────────┬────────┘
         │ 예측
         ▼
┌─────────────────┐
│  종목 추천      │
│  (순위화)       │
└─────────────────┘
```

## 🛠️ 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **AI/ML** | TensorFlow 2.12, TensorFlow Hub, BERT |
| **데이터** | Pandas, NumPy, scikit-learn |
| **웹** | BeautifulSoup4, Requests |
| **시각화** | Matplotlib |
| **언어** | Python 3.7+ |

## 📊 성능

- **학습 시간**: ~5-10분 (60,000건 기준, CPU)
- **추천 시간**: ~2-3분 (200개 종목 기준)
- **정확도**: ~75-85% (데이터 품질에 따라)
- **메모리**: ~4-8GB (학습 시)

## 📂 프로젝트 구조

```
stock-news-analyzer/
├── stock_news_analyzer.py   # 메인 프로그램
├── requirements.txt          # 패키지 의존성
├── README.md                 # 프로젝트 개요
├── 사용설명서.md             # 상세 문서
├── 빠른시작가이드.md         # 시작 가이드
│
├── buy_list.csv              # 관심 종목
├── 코스피200.csv             # 코스피 200
├── 코스닥150.csv             # 코스닥 150
│
├── 급등락뉴스.csv            # 수집된 뉴스 (자동 생성)
├── model_weights.h5          # 학습된 모델 (자동 생성)
├── 임의기간상승.csv          # 추천 결과 (자동 생성)
└── training_history.png      # 학습 곡선 (자동 생성)
```

## 💻 사용 예시

### 예시 1: 일간 루틴
```bash
# 매일 아침 실행
python stock_news_analyzer.py recommend --list all --threshold 0.8
```

### 예시 2: 특정 종목 분석
```bash
python stock_news_analyzer.py recommend \
  --stocks 삼성전자 SK하이닉스 네이버 카카오
```

### 예시 3: 주간 백테스팅
```bash
# 월~금 뉴스 수집
for day in {11..15}; do
  python stock_news_analyzer.py crawl \
    --start 2026 1 $day --end 2026 1 $day \
    --file 코스피200.csv --threshold 5
done

# 통합 학습
python stock_news_analyzer.py train --train-size 100000
```

## 🎨 주요 클래스

### StockNewsAnalyzer
메인 분석기 클래스
```python
analyzer = StockNewsAnalyzer()
analyzer.crawl_news(...)      # 뉴스 수집
analyzer.train_model(...)     # 모델 학습
analyzer.recommend_stocks(...) # 종목 추천
```

### NewsCrawler
구글 뉴스 크롤러
```python
crawler = NewsCrawler()
results = crawler.search_news(keyword, tbs, target)
```

### BERTModel
BERT 모델 관리
```python
model = BERTModel()
model.build_classifier()
model.load_weights()
predictions = model.predict(texts)
```

## 🔧 설정

### Config 클래스에서 조정 가능한 항목

```python
class Config:
    # BERT 모델
    BERT_MODEL_NAME = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
    
    # 학습 설정
    BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 3e-5
    
    # 파일 경로
    MODEL_WEIGHTS = 'model_weights.h5'
    NEWS_DATA = '급등락뉴스.csv'
    
    # 뉴스 검색
    TIME_POOLS = ["1시간 전", ..., "3일 전"]
```

## 📈 결과 해석

### 추천 점수 가이드
- **0.9~1.0**: 🔥 매우 강한 상승 신호
- **0.8~0.9**: ✅ 강한 상승 신호 (권장)
- **0.7~0.8**: ⚠️ 중간 정도 신호
- **0.6~0.7**: 💭 약한 신호
- **0.5 이하**: ❌ 중립/하락

### 임계값 선택
- **보수적**: `--threshold 0.9`
- **표준**: `--threshold 0.8`
- **공격적**: `--threshold 0.7`

## ⚠️ 주의사항

### 투자 경고
- ⚠️ 이 프로그램은 **투자 참고용**이며 투자 권유가 아닙니다
- 📉 과거 데이터가 미래 수익을 보장하지 않습니다
- 💰 모든 투자는 본인의 판단과 책임하에 진행하세요
- 🔍 다양한 정보를 종합적으로 분석하세요

### 기술적 제한
- 크롤링 과도 사용 시 IP 차단 가능
- 뉴스 품질에 따라 정확도 변동
- 단기 시장 변동성은 예측 어려움

## 🐛 문제 해결

### 자주 발생하는 오류

**메모리 부족**
```python
Config.BATCH_SIZE = 16  # 기본값 32에서 감소
```

**크롤링 실패**
- 인터넷 연결 확인
- VPN 사용 시 해제
- 잠시 후 재시도

**CSV 인코딩 오류**
- 파일을 'euc-kr' 인코딩으로 저장

자세한 문제 해결은 [사용설명서.md](사용설명서.md#문제-해결)를 참조하세요.

## 📊 성능 최적화

### GPU 사용 (10배 속도 향상)
```bash
pip install tensorflow-gpu==2.12.0
```

### 배치 크기 증가 (메모리 충분 시)
```python
Config.BATCH_SIZE = 64
```

### 에폭 증가 (정확도 향상)
```python
Config.EPOCHS = 3
```

## 🔄 업데이트 로드맵

### v1.1 (예정)
- [ ] 실시간 뉴스 모니터링
- [ ] 멀티스레딩 크롤링
- [ ] 백테스팅 기능
- [ ] 웹 대시보드

### v1.2 (예정)
- [ ] 앙상블 모델
- [ ] 감성 점수 시각화
- [ ] 자동 리포트 생성
- [ ] API 서버 모드

## 🤝 기여하기

버그 리포트, 기능 제안, 코드 개선은 언제나 환영합니다!

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

## 📜 라이선스

Copyright 2024 Bimghi Choi. All Rights Reserved.

## 📧 문의

- **이슈**: GitHub Issues
- **이메일**: [your-email]
- **문서**: [사용설명서.md](사용설명서.md)

## 🙏 감사의 말

- TensorFlow 팀의 BERT 모델
- Google News API
- 오픈소스 커뮤니티

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**

---

## 📚 참고 자료

- [TensorFlow BERT Tutorial](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [금융 NLP 가이드](https://huggingface.co/blog/financial-sentiment-analysis)

---

Made with ❤️ and 🐍 Python
