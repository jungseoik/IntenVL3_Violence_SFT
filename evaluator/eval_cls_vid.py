import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(csv_file_path):
    """
    CSV 파일을 로드하고 전처리합니다.
    
    Args:
        csv_file_path (str): CSV 파일 경로
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    # 데이터 로드
    df = pd.read_csv(csv_file_path)
    
    print("=== 데이터 기본 정보 ===")
    print(f"총 데이터 수: {len(df)}")
    print(f"결측값 현황:")
    print(df.isnull().sum())
    print()
    
    # 결측값 처리 (predicted_category가 null인 경우 제외)
    df_clean = df.dropna(subset=['predicted_category']).copy()
    print(f"전처리 후 데이터 수: {len(df_clean)}")
    
    return df_clean

def normalize_categories_for_binary_classification(df):
    """
    이진분류를 위한 카테고리 정규화 (Normal vs Abnormal).
    Normal이 아니면 모두 Abnormal로 처리합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
    
    Returns:
        pd.DataFrame: 정규화된 데이터프레임
    """
    df = df.copy()
    
    # 대소문자 통일 및 공백 제거
    df['ground_truth_clean'] = df['ground_truth'].str.lower().str.strip()
    df['predicted_category_clean'] = df['predicted_category'].str.lower().str.strip()
    
    # 이진분류: Normal vs Abnormal 변환
    df['ground_truth_binary'] = df['ground_truth_clean'].apply(
        lambda x: 'normal' if x == 'normal' else 'abnormal'
    )
    
    df['predicted_category_binary'] = df['predicted_category_clean'].apply(
        lambda x: 'normal' if x == 'normal' else 'abnormal'
    )
    
    print("=== 이진분류 카테고리 정규화 결과 ===")
    print("Ground Truth 분포:")
    print(df['ground_truth_binary'].value_counts())
    print("\nPredicted 분포:")
    print(df['predicted_category_binary'].value_counts())
    print()
    
    # 원본 카테고리별 변환 결과 확인
    print("=== 원본 → 이진분류 변환 매핑 ===")
    print("Ground Truth 변환:")
    gt_mapping = df.groupby(['ground_truth_clean', 'ground_truth_binary']).size().reset_index()
    gt_mapping.columns = ['original', 'binary', 'count']
    for _, row in gt_mapping.iterrows():
        print(f"  {row['original']} → {row['binary']} ({row['count']}개)")
    
    print("\nPredicted 변환 (상위 10개):")
    pred_mapping = df.groupby(['predicted_category_clean', 'predicted_category_binary']).size().reset_index()
    pred_mapping.columns = ['original', 'binary', 'count']
    pred_mapping = pred_mapping.sort_values('count', ascending=False).head(10)
    for _, row in pred_mapping.iterrows():
        print(f"  '{row['original']}' → {row['binary']} ({row['count']}개)")
    print()
    
    return df

def calculate_accuracy_by_case(df):
    """
    각 케이스별로 정확도를 계산합니다.
    
    Args:
        df (pd.DataFrame): 정규화된 데이터프레임
    
    Returns:
        pd.DataFrame: 케이스별 정확도 결과
    """
    # 정답 여부 계산 (이진분류 기준)
    df['is_correct'] = (df['ground_truth_binary'] == df['predicted_category_binary'])
    
    # 각 케이스별 그룹화
    case_results = df.groupby(['template_type', 'num_segment']).agg({
        'is_correct': ['count', 'sum', 'mean'],
        'video_name': 'nunique'
    }).round(4)
    
    # 컬럼명 정리
    case_results.columns = ['total_count', 'correct_count', 'accuracy', 'unique_videos']
    case_results = case_results.reset_index()
    
    # 정확도를 퍼센트로 변환
    case_results['accuracy_percent'] = (case_results['accuracy'] * 100).round(2)
    
    # 추가 메트릭 계산
    for idx, row in case_results.iterrows():
        template = row['template_type']
        segment = row['num_segment']
        subset = df[(df['template_type'] == template) & (df['num_segment'] == segment)]
        
        # True Positive, False Positive, True Negative, False Negative 계산
        tp = len(subset[(subset['ground_truth_binary'] == 'abnormal') & (subset['predicted_category_binary'] == 'abnormal')])
        fp = len(subset[(subset['ground_truth_binary'] == 'normal') & (subset['predicted_category_binary'] == 'abnormal')])
        tn = len(subset[(subset['ground_truth_binary'] == 'normal') & (subset['predicted_category_binary'] == 'normal')])
        fn = len(subset[(subset['ground_truth_binary'] == 'abnormal') & (subset['predicted_category_binary'] == 'normal')])
        
        case_results.loc[idx, 'true_positive'] = tp
        case_results.loc[idx, 'false_positive'] = fp
        case_results.loc[idx, 'true_negative'] = tn
        case_results.loc[idx, 'false_negative'] = fn
        
        # Precision, Recall, F1-score 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        case_results.loc[idx, 'precision'] = round(precision, 4)
        case_results.loc[idx, 'recall'] = round(recall, 4)
        case_results.loc[idx, 'f1_score'] = round(f1, 4)
    
    return case_results

def detailed_performance_analysis(df):
    """
    상세한 성능 분석을 수행합니다.
    
    Args:
        df (pd.DataFrame): 정규화된 데이터프레임
    
    Returns:
        dict: 분석 결과 딕셔너리
    """
    results = {}
    
    # 전체 성능
    overall_accuracy = (df['ground_truth_binary'] == df['predicted_category_binary']).mean()
    results['overall_accuracy'] = overall_accuracy
    
    # 템플릿별 성능
    template_performance = df.groupby('template_type').apply(
        lambda x: (x['ground_truth_binary'] == x['predicted_category_binary']).mean()
    )
    results['template_performance'] = template_performance
    
    # num_segment별 성능
    segment_performance = df.groupby('num_segment').apply(
        lambda x: (x['ground_truth_binary'] == x['predicted_category_binary']).mean()
    )
    results['segment_performance'] = segment_performance
    
    return results



def analyze_error_patterns(df):
    """
    오답 패턴을 분석합니다.
    
    Args:
        df (pd.DataFrame): 정규화된 데이터프레임
    
    Returns:
        pd.DataFrame: 오답 패턴 분석 결과
    """
    # 오답인 경우만 필터링 (이진분류 기준)
    errors = df[df['ground_truth_binary'] != df['predicted_category_binary']].copy()
    
    if len(errors) == 0:
        print("오답이 없습니다!")
        return pd.DataFrame()
    
    # 이진분류 오답 패턴 분석
    print("=== 이진분류 오답 패턴 ===")
    binary_error_patterns = errors.groupby(['ground_truth_binary', 'predicted_category_binary']).size().reset_index()
    binary_error_patterns.columns = ['ground_truth', 'predicted', 'error_count']
    binary_error_patterns = binary_error_patterns.sort_values('error_count', ascending=False)
    print(binary_error_patterns.to_string(index=False))
    print()
    
    # 원본 카테고리 오답 패턴도 확인
    print("=== 원본 카테고리 오답 패턴 (상위 10개) ===")
    original_error_patterns = errors.groupby(['ground_truth_clean', 'predicted_category_clean']).size().reset_index()
    original_error_patterns.columns = ['ground_truth_original', 'predicted_original', 'error_count']
    original_error_patterns = original_error_patterns.sort_values('error_count', ascending=False).head(10)
    print(original_error_patterns.to_string(index=False))
    print()
    
    # 케이스별 오답률
    case_error_rates = errors.groupby(['template_type', 'num_segment']).size().reset_index()
    case_error_rates.columns = ['template_type', 'num_segment', 'error_count']
    
    # 전체 케이스별 데이터 수와 병합
    case_totals = df.groupby(['template_type', 'num_segment']).size().reset_index()
    case_totals.columns = ['template_type', 'num_segment', 'total_count']
    
    case_analysis = pd.merge(case_totals, case_error_rates, on=['template_type', 'num_segment'], how='left')
    case_analysis['error_count'] = case_analysis['error_count'].fillna(0)
    case_analysis['error_rate'] = (case_analysis['error_count'] / case_analysis['total_count'] * 100).round(2)
    
    return case_analysis

def generate_comprehensive_report(csv_file_path):
    """
    종합적인 평가 리포트를 생성합니다.
    
    Args:
        csv_file_path (str): CSV 파일 경로
    """
    print("🔍 모델 성능 평가 시작")
    print("=" * 60)
    
    # 1. 데이터 로드 및 전처리
    df = load_and_preprocess_data(csv_file_path)
    df = normalize_categories_for_binary_classification(df)
    
    # 2. 케이스별 정확도 계산
    print("📊 케이스별 성능 분석 (이진분류)")
    print("=" * 60)
    case_results = calculate_accuracy_by_case(df)
    
    # 주요 메트릭만 출력
    display_columns = ['template_type', 'num_segment', 'accuracy_percent', 'precision', 'recall', 'f1_score', 
                      'true_positive', 'false_positive', 'true_negative', 'false_negative']
    print(case_results[display_columns].to_string(index=False))
    print()
    
    # 3. 상세 성능 분석
    print("📈 상세 성능 분석")
    print("=" * 60)
    detailed_results = detailed_performance_analysis(df)
    
    print(f"전체 정확도: {detailed_results['overall_accuracy']:.4f} ({detailed_results['overall_accuracy']*100:.2f}%)")
    print()
    
    print("템플릿별 성능:")
    for template, accuracy in detailed_results['template_performance'].items():
        print(f"  {template}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    print("세그먼트별 성능:")
    for segment, accuracy in detailed_results['segment_performance'].items():
        print(f"  {segment} segments: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # 4. 오답 패턴 분석
    print("❌ 오답 패턴 분석")
    print("=" * 60)
    error_analysis = analyze_error_patterns(df)
    if not error_analysis.empty:
        print("케이스별 오답 현황:")
        print(error_analysis.to_string(index=False))
    print()
    
    # 5. 최고 성능 케이스 찾기
    print("🏆 최고 성능 케이스")
    print("=" * 60)
    best_case = case_results.loc[case_results['accuracy'].idxmax()]
    print(f"최고 성능: {best_case['template_type']} + {best_case['num_segment']} segments")
    print(f"정확도: {best_case['accuracy']:.4f} ({best_case['accuracy_percent']:.2f}%)")
    print(f"정답 수: {best_case['correct_count']}/{best_case['total_count']}")
    print()
    
    print("✅ 평가 완료!")
    
    return df, case_results


if __name__ == "__main__":
    csv_file_path = "results.csv"  # 실제 파일 경로로 변경하세요
    df, case_results = generate_comprehensive_report(csv_file_path)
