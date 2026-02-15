#!/bin/bash
# 예제 실행 스크립트
# BERT 기반 주식 뉴스 분석 시스템

echo "=========================================="
echo "BERT 주식 뉴스 분석 시스템 - 예제 실행"
echo "=========================================="
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수: 헤더 출력
print_header() {
    echo ""
    echo "${GREEN}=========================================="
    echo "$1"
    echo "==========================================${NC}"
    echo ""
}

# 함수: 경고 메시지
print_warning() {
    echo "${YELLOW}⚠️  $1${NC}"
}

# 함수: 성공 메시지
print_success() {
    echo "${GREEN}✓ $1${NC}"
}

# 함수: 에러 메시지
print_error() {
    echo "${RED}✗ $1${NC}"
}

# 함수: 파일 존재 확인
check_file() {
    if [ -f "$1" ]; then
        print_success "파일 확인: $1"
        return 0
    else
        print_error "파일 없음: $1"
        return 1
    fi
}

# 예제 선택 메뉴
show_menu() {
    print_header "예제 선택"
    echo "1. 전체 워크플로우 (추천)"
    echo "2. 뉴스 크롤링만"
    echo "3. 모델 학습만"
    echo "4. 종목 추천만"
    echo "5. 개별 종목 분석"
    echo "6. 자동화 스크립트 생성"
    echo "0. 종료"
    echo ""
    read -p "선택 (0-6): " choice
    echo ""
    return $choice
}

# 예제 1: 전체 워크플로우
example_full_workflow() {
    print_header "예제 1: 전체 워크플로우"
    
    print_warning "이 예제는 15~20분 정도 소요됩니다."
    read -p "계속하시겠습니까? (y/n): " confirm
    
    if [ "$confirm" != "y" ]; then
        echo "취소되었습니다."
        return
    fi
    
    # Step 1: 뉴스 크롤링
    print_header "Step 1/3: 뉴스 크롤링"
    echo "전날 10% 이상 상승한 종목의 뉴스를 수집합니다..."
    
    python stock_news_analyzer.py crawl \
        --start 2026 1 15 \
        --end 2026 1 15 \
        --file buy_list.csv \
        --threshold 10
    
    if [ $? -eq 0 ]; then
        print_success "뉴스 크롤링 완료"
        check_file "급등락뉴스.csv"
    else
        print_error "뉴스 크롤링 실패"
        return
    fi
    
    # Step 2: 모델 학습
    print_header "Step 2/3: 모델 학습"
    echo "BERT 모델을 학습합니다..."
    
    python stock_news_analyzer.py train
    
    if [ $? -eq 0 ]; then
        print_success "모델 학습 완료"
        check_file "model_weights.h5"
        check_file "training_history.png"
    else
        print_error "모델 학습 실패"
        return
    fi
    
    # Step 3: 종목 추천
    print_header "Step 3/3: 종목 추천"
    echo "학습된 모델로 종목을 추천합니다..."
    
    python stock_news_analyzer.py recommend \
        --list all \
        --threshold 0.8
    
    if [ $? -eq 0 ]; then
        print_success "종목 추천 완료"
        check_file "임의기간상승.csv"
        
        echo ""
        echo "========== 추천 종목 Top 5 =========="
        head -n 6 임의기간상승.csv | tail -n 5
        echo "===================================="
    else
        print_error "종목 추천 실패"
        return
    fi
    
    print_success "전체 워크플로우 완료!"
}

# 예제 2: 뉴스 크롤링만
example_crawl_only() {
    print_header "예제 2: 뉴스 크롤링"
    
    echo "크롤링 옵션:"
    echo "1. 상승 종목 (10% 이상)"
    echo "2. 하락 종목 (5% 이상)"
    echo "3. 커스텀 설정"
    read -p "선택 (1-3): " crawl_option
    
    case $crawl_option in
        1)
            python stock_news_analyzer.py crawl \
                --start 2026 1 15 \
                --end 2026 1 15 \
                --file buy_list.csv \
                --threshold 10
            ;;
        2)
            python stock_news_analyzer.py crawl \
                --start 2026 1 15 \
                --end 2026 1 15 \
                --file buy_list.csv \
                --threshold -5
            ;;
        3)
            read -p "시작 년: " y1
            read -p "시작 월: " m1
            read -p "시작 일: " d1
            read -p "종료 년: " y2
            read -p "종료 월: " m2
            read -p "종료 일: " d2
            read -p "종목 파일: " file
            read -p "임계값: " threshold
            
            python stock_news_analyzer.py crawl \
                --start $y1 $m1 $d1 \
                --end $y2 $m2 $d2 \
                --file $file \
                --threshold $threshold
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        print_success "크롤링 완료"
    else
        print_error "크롤링 실패"
    fi
}

# 예제 3: 모델 학습만
example_train_only() {
    print_header "예제 3: 모델 학습"
    
    if ! check_file "급등락뉴스.csv"; then
        print_error "학습 데이터가 없습니다. 먼저 뉴스를 크롤링하세요."
        return
    fi
    
    # 데이터 행 수 확인
    data_count=$(wc -l < 급등락뉴스.csv)
    echo "학습 데이터: $data_count 건"
    
    read -p "학습 데이터 크기 (기본 60000): " train_size
    train_size=${train_size:-60000}
    
    if [ $data_count -lt $train_size ]; then
        print_warning "데이터가 부족합니다. 가능한 크기로 조정합니다."
        train_size=$((data_count - 1000))
    fi
    
    python stock_news_analyzer.py train --train-size $train_size
    
    if [ $? -eq 0 ]; then
        print_success "학습 완료"
        check_file "model_weights.h5"
        
        if [ -f "training_history.png" ]; then
            print_success "학습 곡선이 저장되었습니다: training_history.png"
        fi
    else
        print_error "학습 실패"
    fi
}

# 예제 4: 종목 추천만
example_recommend_only() {
    print_header "예제 4: 종목 추천"
    
    if ! check_file "model_weights.h5"; then
        print_error "학습된 모델이 없습니다. 먼저 모델을 학습하세요."
        return
    fi
    
    echo "추천 옵션:"
    echo "1. 관심 종목 (buy_list.csv)"
    echo "2. 코스피 200"
    echo "3. 코스닥 150"
    echo "4. 전체 종목"
    read -p "선택 (1-4): " list_option
    
    case $list_option in
        1) list="buy" ;;
        2) list="kospi" ;;
        3) list="kosdaq" ;;
        4) list="all" ;;
    esac
    
    read -p "추천 임계값 (0.0~1.0, 기본 0.8): " threshold
    threshold=${threshold:-0.8}
    
    python stock_news_analyzer.py recommend \
        --list $list \
        --threshold $threshold
    
    if [ $? -eq 0 ]; then
        print_success "추천 완료"
        
        echo ""
        echo "========== 추천 결과 =========="
        head -n 11 임의기간상승.csv
        echo "=============================="
    else
        print_error "추천 실패"
    fi
}

# 예제 5: 개별 종목 분석
example_individual() {
    print_header "예제 5: 개별 종목 분석"
    
    if ! check_file "model_weights.h5"; then
        print_error "학습된 모델이 없습니다. 먼저 모델을 학습하세요."
        return
    fi
    
    echo "분석할 종목을 입력하세요 (띄어쓰기로 구분):"
    echo "예: 삼성전자 SK하이닉스 네이버"
    read -p "종목: " stocks
    
    if [ -z "$stocks" ]; then
        print_error "종목을 입력하지 않았습니다."
        return
    fi
    
    python stock_news_analyzer.py recommend \
        --stocks $stocks \
        --threshold 0.0
    
    if [ $? -eq 0 ]; then
        print_success "분석 완료"
        
        echo ""
        echo "========== 분석 결과 =========="
        tail -n $(($(echo $stocks | wc -w) + 1)) 임의기간상승.csv
        echo "=============================="
    else
        print_error "분석 실패"
    fi
}

# 예제 6: 자동화 스크립트 생성
example_automation() {
    print_header "예제 6: 자동화 스크립트 생성"
    
    cat > daily_analysis.sh << 'EOF'
#!/bin/bash
# 일일 자동 분석 스크립트

DATE=$(date +%Y-%m-%d)
LOG_FILE="logs/analysis_${DATE}.log"

mkdir -p logs

{
    echo "=========================================="
    echo "일일 분석 시작: $DATE"
    echo "=========================================="
    
    # 전날 뉴스 수집
    echo "[1/3] 뉴스 수집 중..."
    python stock_news_analyzer.py crawl \
        --start $(date -d "yesterday" +%Y) $(date -d "yesterday" +%m) $(date -d "yesterday" +%d) \
        --end $(date -d "yesterday" +%Y) $(date -d "yesterday" +%m) $(date -d "yesterday" +%d) \
        --file buy_list.csv \
        --threshold 5
    
    # 금요일이면 재학습
    if [ $(date +%u) -eq 5 ]; then
        echo "[2/3] 주간 모델 재학습..."
        python stock_news_analyzer.py train
    else
        echo "[2/3] 재학습 생략 (금요일에만 실행)"
    fi
    
    # 종목 추천
    echo "[3/3] 종목 추천..."
    python stock_news_analyzer.py recommend \
        --list all \
        --threshold 0.8 \
        --output "recommendations_${DATE}.csv"
    
    echo ""
    echo "========== 오늘의 추천 종목 =========="
    head -n 6 "recommendations_${DATE}.csv"
    echo "===================================="
    
    echo ""
    echo "분석 완료: $DATE"
    
} 2>&1 | tee "$LOG_FILE"
EOF
    
    chmod +x daily_analysis.sh
    print_success "자동화 스크립트 생성 완료: daily_analysis.sh"
    
    echo ""
    echo "크론탭에 등록하려면 다음 명령을 실행하세요:"
    echo "${YELLOW}crontab -e${NC}"
    echo ""
    echo "그리고 다음 라인을 추가하세요 (매일 오전 9시 실행):"
    echo "${YELLOW}0 9 * * * $(pwd)/daily_analysis.sh${NC}"
}

# 메인 루프
main() {
    while true; do
        show_menu
        choice=$?
        
        case $choice in
            1) example_full_workflow ;;
            2) example_crawl_only ;;
            3) example_train_only ;;
            4) example_recommend_only ;;
            5) example_individual ;;
            6) example_automation ;;
            0) 
                echo "종료합니다."
                exit 0
                ;;
            *)
                print_error "잘못된 선택입니다."
                ;;
        esac
        
        echo ""
        read -p "계속하려면 Enter를 누르세요..."
    done
}

# 초기 확인
print_header "시작 전 확인"

# Python 확인
if command -v python &> /dev/null; then
    python_version=$(python --version 2>&1)
    print_success "Python 설치됨: $python_version"
else
    print_error "Python이 설치되지 않았습니다."
    exit 1
fi

# 필수 파일 확인
print_header "필수 파일 확인"
check_file "stock_news_analyzer.py"
check_file "buy_list.csv"
check_file "코스피200.csv"
check_file "코스닥150.csv"

echo ""
read -p "시작하려면 Enter를 누르세요..."

# 메인 실행
main
