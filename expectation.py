# -*- coding:utf-8 -*-

# 경고 라이브러리
import warnings

# re
import re

# 기본 라이브러리
import pandas as pd
import numpy as np

# GeoPandas 관련
import geopandas as gpd

# 데이터 분할용
from sklearn.model_selection import train_test_split

# 머신 평가용
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# 스케일링 관련
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler

# 모델
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

# 하이퍼파라미터 튜닝용
from sklearn.model_selection import GridSearchCV

# 스트림릿
import streamlit as st

# forlium 시각화
import folium
from folium.plugins import MarkerCluster

# 그래프 시각화
import plotly.graph_objs as go

# 폰트 매니저
from matplotlib import font_manager, rc

font_path = r"font/NanumGothic-Regular.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# ----------------------------- # --------------(이하 함수 코드)--------------- # ----------------------------- #


# 데이터 설정
@st.cache_data
def get_data():
    data = pd.read_csv('data2/final_merged_update_store_age_df.csv')

    # 데이터 전처리
    # 새로운 피처 추가
    data['편의점_밀도'] = data['유사_업종_점포_수']/data['영역_면적']

    # 불필요한 컬럼 제거
    data = data.drop(columns=['영역_면적', '기준_년','상권_코드', '자치구_코드', '행정동_코드', '자치구_코드_명', '평일_유동인구_수_평균', '주말_유동인구_수_평균', '소득_구간_코드', '점포_수', '개업_율', '폐업_률', '프랜차이즈_점포_수'])
    return data


# 데이터 설정
@st.cache_data
def set_data(data):
    # 범주형 변수와 숫자형 변수 구분
    cat_cols = ['기준_분기', '시간대', '상권_구분_코드_명', '상권_코드_명', '행정동_코드_명']
    num_cols = data.columns.difference(cat_cols).tolist()

    # 독립변수와 종속변수 분리
    y = data.pop('시간대_매출금액')
    X = data

    # 독립변수 데이터 스케일링
    # 변수 구분 코드에서 종속변수 제거
    num_cols.remove('시간대_매출금액')

    ## 범주형 변수 더미화
    X = pd.get_dummies(X, columns=cat_cols)

    ## 더미 변수화된 값이 불리언 형태로 나왔다면 0과 1로 변환
    X.replace({True: 1, False: 0}, inplace=True)

    ## 숫자형 변수 정규화
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y

# 학습 데이터와 테스트 데이터로 분할
def train_test_division(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)
    return X_train, X_test, y_train, y_test

# 피처 이름 정제
def clean_feature_names(X):
    # 특수 문자를 제거하고 언더스코어(_)로 대체합니다.
    X.columns = [re.sub(r'\W+', '_', col) for col in X.columns]
    return X

# feature 컬럼명 정제
def refine_feature_name(X_train, X_test):
    # 데이터프레임 X의 특성 이름을 정제합니다.
    X_train = clean_feature_names(X_train)
    X_test = clean_feature_names(X_test)
    return X_train, X_test

# box-cox 변환
def box_cox(y_train, y_test):
    y_train_boxcox, lambda_ = boxcox(y_train + 1)  # +1을 더하는 이유는 음수 값이 없도록 하기 위함
    y_test_boxcox = boxcox(y_test + 1, lmbda=lambda_)
    return y_train_boxcox, y_test_boxcox, lambda_

# 모델 생성 및 훈련
@st.cache_data
def train_model(X_train, y_train_boxcox):
    # 모델 학습 - LightGBM
    lgbm_regression = LGBMRegressor(random_state=11, verbose=-1)

    # 탐색할 하이퍼파라미터 범위 설정
    param_grid = {
        'num_leaves': [15, 31, 50],
        'learning_rate': [0.1, 0.15, 0.2],
        'n_estimators': [200, 300, 400]
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(estimator=lgbm_regression, param_grid=param_grid, cv=3)

    # 모델 학습
    grid_search.fit(X_train, y_train_boxcox)

    # 최적의 estimator
    best_lgbm_regression = grid_search.best_estimator_
    return best_lgbm_regression

# 모델 예측
def eveluation(best_lgbm_regression, X_test, lambda_):
    y_pred_lgbm_grid_boxcox = best_lgbm_regression.predict(X_test)

    # 예측 결과를 원래의 스케일로 되돌리기 위해 역 Box-Cox 변환 적용
    y_pred_lgbm_grid = inv_boxcox(y_pred_lgbm_grid_boxcox, lambda_)

    return y_pred_lgbm_grid

# ----------------------------- # --------------(이하 시각화를 위한 예측 테이블 코드)--------------- # ----------------------------- #


# 먼저 시각화에 활용할 베이스 데이터 테이블이 필요하다.

@st.cache_data
def get_base_data():
    # 데이터 불러오기
    base_data = pd.read_csv('data2/final_merged_update_store_age_df.csv')

    # 새로운 피처 추가
    base_data['편의점_밀도'] = base_data['유사_업종_점포_수']/base_data['영역_면적']

    # 불필요한 컬럼 제거
    base_data = base_data.drop(columns=['영역_면적', '기준_년','상권_코드', '자치구_코드', '행정동_코드', '자치구_코드_명', '평일_유동인구_수_평균', '주말_유동인구_수_평균', '소득_구간_코드', '점포_수', '개업_율', '폐업_률', '프랜차이즈_점포_수'])
    return base_data

# 모델 학습을 위한 데이터 전처리
def get_y_table_boxcox(base_data):
    # 베이스 데이터 유지를 위해, 데이터를 카피
    data_a = base_data.copy()

    # 범주형 변수와 숫자형 변수 구분
    cat_cols = ['기준_분기', '시간대', '상권_구분_코드_명', '상권_코드_명', '행정동_코드_명']
    num_cols = data_a.columns.difference(cat_cols).tolist()


    # 독립변수와 종속변수 분리
    y_table = data_a.pop('시간대_매출금액')
    X_table = data_a

    # 독립변수 데이터 스케일링
    # 변수 구분 코드에서 종속변수 제거
    num_cols.remove('시간대_매출금액')

    ## 범주형 변수 더미화
    X_table = pd.get_dummies(X_table, columns=cat_cols)

    ## 더미 변수화된 값이 불리언 형태로 나왔다면 0과 1로 변환
    X_table.replace({True: 1, False: 0}, inplace=True)

    ## 숫자형 변수 정규화
    scaler = StandardScaler()
    X_table[num_cols] = scaler.fit_transform(X_table[num_cols])


    def clean_feature_names(X):
        # 특수 문자를 제거하고 언더스코어(_)로 대체합니다.
        X.columns = [re.sub(r'\W+', '_', col) for col in X.columns]
        return X

    # 데이터프레임 X의 특성 이름을 정제합니다.
    X_table = clean_feature_names(X_table)


    # Box-Cox 변환 적용
    y_table_boxcox, lambda_ = boxcox(y_table + 1)  # +1을 더하는 이유는 음수 값이 없도록 하기 위함

    return X_table, y_table_boxcox, lambda_

# 테이블용 예측
def pred_for_table(best_lgbm_regression, X_table, lambda_):
    # X_table에 대한 매출 예측
    y_ForTable_pred_lgbm_grid_boxcox = best_lgbm_regression.predict(X_table)

    # 결과를 원래 단위로 돌리기 위해 역 박스콕스 변환
    y_ForTable_pred_lgbm_grid = inv_boxcox(y_ForTable_pred_lgbm_grid_boxcox, lambda_)

    return y_ForTable_pred_lgbm_grid

# 베이스 데이터로 테이블 만들기2 : 또 다른 전처리
def get_base_data_2(base_data, y_ForTable_pred_lgbm_grid):
    # base_data에서 '시간대_매출금액' 제거
    base_data = base_data.drop(columns=['시간대_매출금액'])

    # '시간대_예상매출' 컬럼 추가
    base_data['시간대_예상매출'] = y_ForTable_pred_lgbm_grid
    return base_data

# 그룹연산 및 좌표 merge : 최종 테이블
# 출력값이 두 개다.
@st.cache_data
def get_groupby_base_data(base_data):
    # base 데이터에서 grupby
    # 기준 : '상권_코드_명', '행정동_코드_명'
    # 대상 : '시간대_예상매출'
    # 방법 : mean
    groupby_base_data = base_data.groupby(['상권_코드_명', '행정동_코드_명', '기준_분기', '시간대'])['시간대_예상매출'].agg(['mean']).reset_index()

    # folium용 좌표 데이터 불러오기
    district_point_df = pd.read_csv('data2/district_centerpoint_for_Folium.csv')

    # district_point_df와 groupby_base_data를 merge
    # 기준 : '상권_코드_명', '행정동_코드_명'

    expected_income_base_df = groupby_base_data.merge(district_point_df, on=['상권_코드_명', '행정동_코드_명'])
    expected_income_base_df = expected_income_base_df.rename(columns={'mean': '시간대_예상매출'}, inplace=False)

    # 매출 단위 변경 : 백만원
    expected_income_base_df['시간대_예상매출'] = expected_income_base_df['시간대_예상매출']/(1e+6)

    return expected_income_base_df, district_point_df


# ----------------------------- # --------------(시각화 구현 순서)--------------- # ----------------------------- #

# 시각화 1-1 : 분기별, 시간대별, 지정 상권 매출 비교 그래프 : 막대 그래프
# 시각화 1-2 : 분기별, 시간대별, 지정 상권 folium 시각화

# 시각화 2 : 전체 상권의 시간별, 분기별 예상매출 확인 : folium

# 시각화 3 : 전체 상권의 예상 연 매출 확인

# 시각화 4 : 전체 상권의 평균 월 매출 확인

# 시각화 5 : 예상 월 매출을 만족하는 지점 찾기

# ----------------------------- # --------------(이하 시각화 코드)--------------- # ----------------------------- #


# ----------------------------- # --------------(시각화 1)--------------- # ----------------------------- #

# 시각화 1-1 : 분기별, 시간대별, 지정 상권 매출 비교 그래프 : 막대 그래프

# 슬라이싱-1
def get_sliced_EIBF_for_1_1(quarter, district, expected_income_base_df):
    # 시간별, 분기별 조건에 따른 슬라이싱

    # 슬라이싱 : 시간대별 비교 그래프용
    sliced_EIBF_for_1_1 = expected_income_base_df.loc[(expected_income_base_df['기준_분기']==quarter)&(expected_income_base_df['상권_코드']==district)].reset_index(drop=True)
    # sliced_EIBF_for_1_1

    return sliced_EIBF_for_1_1

# 슬라이싱-2
def get_sliced_EIBF_for_1_2(quarter, time, district, expected_income_base_df):
    # 시간별, 분기별 조건에 따른 슬라이싱

    # 슬라이싱 : 지도 시각화용
    sliced_EIBF_for_1_2 = expected_income_base_df.loc[(expected_income_base_df['기준_분기']==quarter)&(expected_income_base_df['시간대']==time)&(expected_income_base_df['상권_코드']==district)].reset_index(drop=True)
    # sliced_EIBF_for_1_2

    return sliced_EIBF_for_1_2

# 시각화
def viz_1_1(quarter, sliced_EIBF_for_1_1):
    # 막대 그래프 생성
    bar_chart = go.Bar(
        x=sliced_EIBF_for_1_1['시간대'],  # x축 데이터
        y=sliced_EIBF_for_1_1['시간대_예상매출'],  # y축 데이터
        text=sliced_EIBF_for_1_1['시간대_예상매출'].round(2),  # 막대 위에 소수점 2째자리까지 데이터 수치 표시
        textposition='outside',  # 막대 위쪽에 데이터 수치 표시
        hoverinfo='text',  # 마우스를 올렸을 때 나타나는 정보 설정
        hovertext='시간대: ' + sliced_EIBF_for_1_1['시간대'].astype(str) + '<br>예상매출(단위:백만원): ' + sliced_EIBF_for_1_1['시간대_예상매출'].round(2).astype(str)  # 막대 위에 표시될 문구 설정
    )

    # 그래프 레이아웃 설정
    layout = go.Layout(
        title=f'{quarter}분기 시간대별 예상 매출 비교',  # 그래프 제목
        xaxis=dict(title='시간대'),  # x축 레이블 설정
        yaxis=dict(title='예상 매출'),  # y축 레이블 설정
        annotations=[dict(  # 그래프 한 쪽에 단위 표시
            text='단위 : 백만원',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.98,
            y=0.98
        )],
        height=600  # 시각화 공간의 높이 설정 (기본값은 450)
    )

    # 그래프 객체 생성
    fig = go.Figure(data=[bar_chart], layout=layout)

    # 그래프 출력
    return fig


# 시각화 1-2 : 분기별, 시간대별, 지정 상권 folium 시각화

# 시각화 : folium
def viz_1_2(sliced_EIBF_for_1_2):
    # 강남구 중심 좌표
    map_center = [37.5172, 127.0473]

    # Folium 맵 생성
    m = folium.Map(location=map_center, zoom_start=13)

    # MarkerCluster 레이어 생성
    marker_cluster = MarkerCluster().add_to(m)

    # 각 점에 대한 정보를 Folium으로 추가
    for idx, row in sliced_EIBF_for_1_2.iterrows():
        sales_amount = round(row['시간대_예상매출'], 2)
        popup_text = f"상권명: {row['상권_코드_명']}</br>행정동: {row['행정동_코드_명']}</br>시간대 예상 매출금액(단위:백만): {sales_amount}"
        popup = folium.map.Popup(popup_text, max_width=200)
        folium.Marker([row['latitude'], row['longitude']], popup=popup, max_width=100).add_to(marker_cluster)

    # 추가사항
    # Folium 맵을 HTML iframe 요소로 래핑하여 출력 : 이것이 없으면 스트림릿에서 출력이 안됨.
    folium_map_html = m.get_root().render()
    st.components.v1.html(folium_map_html, width=700, height=500)

    # Streamlit에 Folium 맵 표시
    m


# ----------------------------- # --------------(시각화 2)--------------- # ----------------------------- #

# 시각화 2 : 전체 상권의 시간별, 분기별 예상매출 확인 : folium

# 슬라이싱
def get_sliced_EIBF_for_2(quarter, time, expected_income_base_df):
    # 슬라이싱
    sliced_EIBF_for_2 = expected_income_base_df.loc[(expected_income_base_df['기준_분기']==quarter)&(expected_income_base_df['시간대']==time)].reset_index(drop=True)
    return sliced_EIBF_for_2

# 시각화 : folium
def viz_2(sliced_EIBF_for_2):
    # 강남구 중심 좌표
    map_center = [37.5172, 127.0473]

    # Folium 맵 생성
    m = folium.Map(location=map_center, zoom_start=13)

    # MarkerCluster 레이어 생성
    marker_cluster = MarkerCluster().add_to(m)

    # 각 점에 대한 정보를 Folium으로 추가
    for idx, row in sliced_EIBF_for_2.iterrows():
        sales_amount = round(row['시간대_예상매출'], 2)
        popup_text = f"상권명: {row['상권_코드_명']}</br>행정동: {row['행정동_코드_명']}</br>시간대 예상 매출금액(단위:백만): {sales_amount}"
        popup = folium.map.Popup(popup_text, max_width=200)
        folium.Marker([row['latitude'], row['longitude']], popup=popup, max_width=100).add_to(marker_cluster)

    # 추가사항
    # Folium 맵을 HTML iframe 요소로 래핑하여 출력 : 이것이 없으면 스트림릿에서 출력이 안됨.
    folium_map_html = m.get_root().render()
    st.components.v1.html(folium_map_html, width=700, height=500)

    # Streamlit에 Folium 맵 표시
    m


# ----------------------------- # --------------(시각화 3)--------------- # ----------------------------- #
    
# 시각화 3 : 전체 상권의 예상 연 매출 확인
    
# 슬라이싱

def get_sliced_EIBF_for_3(expected_income_base_df, district_point_df):
    # 그룹바이 적용으로 연 매출 구하기.
    sliced_EIBF_for_3 = expected_income_base_df.groupby(['상권_코드_명', '행정동_코드_명'])['시간대_예상매출'].agg(['sum']).reset_index()
    sliced_EIBF_for_3 = sliced_EIBF_for_3.rename(columns={'sum': '예상_연매출'}, inplace=False)

    # 좌표 합치기
    sliced_EIBF_for_3 = sliced_EIBF_for_3.merge(district_point_df, on=['상권_코드_명', '행정동_코드_명'])
    return sliced_EIBF_for_3


# 시각화 : folium
def viz_3(sliced_EIBF_for_3):
    # 강남구 중심 좌표
    map_center = [37.5172, 127.0473]

    # Folium 맵 생성
    m = folium.Map(location=map_center, zoom_start=13)

    # MarkerCluster 레이어 생성
    marker_cluster = MarkerCluster().add_to(m)

    # 각 점에 대한 정보를 Folium으로 추가
    for idx, row in sliced_EIBF_for_3.iterrows():
        sales_amount = round(row['예상_연매출'], 2)
        popup_text = f"상권명: {row['상권_코드_명']}</br>행정동: {row['행정동_코드_명']}</br>예상 연 매출 금액(단위:백만): {sales_amount}"
        popup = folium.map.Popup(popup_text, max_width=200)
        folium.Marker([row['latitude'], row['longitude']], popup=popup, max_width=100).add_to(marker_cluster)

    # 추가사항
    # Folium 맵을 HTML iframe 요소로 래핑하여 출력 : 이것이 없으면 스트림릿에서 출력이 안됨.
    folium_map_html = m.get_root().render()
    st.components.v1.html(folium_map_html, width=700, height=500)

    # Streamlit에 Folium 맵 표시
    m


# ----------------------------- # --------------(시각화 4)--------------- # ----------------------------- #
    
# 시각화 4 : 전체 상권의 평균 월 매출 확인
    
# 전처리
def get_sliced_EIBF_for_4(sliced_EIBF_for_3):
    # 월 평균 매출 계산 : 연매출/12
    sliced_EIBF_for_4 = sliced_EIBF_for_3.copy()
    sliced_EIBF_for_4['예상_월매출'] = sliced_EIBF_for_4['예상_연매출']/12
    sliced_EIBF_for_4 = sliced_EIBF_for_4.drop(columns=['예상_연매출'])
    return sliced_EIBF_for_4

# 시각화 : folium
def viz_4(sliced_EIBF_for_4):
    # 강남구 중심 좌표
    map_center = [37.5172, 127.0473]

    # Folium 맵 생성
    m = folium.Map(location=map_center, zoom_start=13)

    # MarkerCluster 레이어 생성
    marker_cluster = MarkerCluster().add_to(m)

    # 각 점에 대한 정보를 Folium으로 추가
    for idx, row in sliced_EIBF_for_4.iterrows():
        sales_amount = round(row['예상_월매출'], 2)
        popup_text = f"상권명: {row['상권_코드_명']}</br>행정동: {row['행정동_코드_명']}</br>예상 월 매출 금액(단위:백만): {sales_amount}"
        popup = folium.map.Popup(popup_text, max_width=200)
        folium.Marker([row['latitude'], row['longitude']], popup=popup, max_width=100).add_to(marker_cluster)

    # 추가사항
    # Folium 맵을 HTML iframe 요소로 래핑하여 출력 : 이것이 없으면 스트림릿에서 출력이 안됨.
    folium_map_html = m.get_root().render()
    st.components.v1.html(folium_map_html, width=700, height=500)

    # Streamlit에 Folium 맵 표시
    m


# ----------------------------- # --------------(시각화 5)--------------- # ----------------------------- #

# 시각화 5 : 예상 월 매출을 만족하는 지점 찾기

# 조건에 따라 슬라이싱
def get_sliced_EIBF_for_5(wish_income_min, wish_income_max, sliced_EIBF_for_4):
    sliced_EIBF_for_5 = sliced_EIBF_for_4.loc[(sliced_EIBF_for_4['예상_월매출']<=wish_income_max)&(sliced_EIBF_for_4['예상_월매출']>=wish_income_min)].reset_index(drop=True)
    return sliced_EIBF_for_5

# 시각화 : folium
def viz_5(sliced_EIBF_for_5):
    # 강남구 중심 좌표
    map_center = [37.5172, 127.0473]

    # Folium 맵 생성
    m = folium.Map(location=map_center, zoom_start=13)

    # MarkerCluster 레이어 생성
    marker_cluster = MarkerCluster().add_to(m)

    # 각 점에 대한 정보를 Folium으로 추가
    for idx, row in sliced_EIBF_for_5.iterrows():
        sales_amount = round(row['예상_월매출'], 2)
        popup_text = f"상권명: {row['상권_코드_명']}</br>행정동: {row['행정동_코드_명']}</br>예상 월 매출 금액(단위:백만): {sales_amount}"
        popup = folium.map.Popup(popup_text, max_width=200)
        folium.Marker([row['latitude'], row['longitude']], popup=popup, max_width=100).add_to(marker_cluster)

    # 추가사항
    # Folium 맵을 HTML iframe 요소로 래핑하여 출력 : 이것이 없으면 스트림릿에서 출력이 안됨.
    folium_map_html = m.get_root().render()
    st.components.v1.html(folium_map_html, width=700, height=500)

    # Streamlit에 Folium 맵 표시
    m


# ----------------------------- # --------------(이하 main)--------------- # ----------------------------- #
    

def expectation_content():

    st.header("강남구 편의점 예상매출 종합")
    st.write('-'*50)

    # 머신러닝을 돌리고, 그에 따른 모델과 테이블을 먼저 얻자.
    # ----------------------------- # --------------(이하 머신러닝)--------------- # ----------------------------- #
    
    # 데이터 불러오기
    data = get_data()

    # 데이터 설정
    X, y = set_data(data)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_division(X, y)

    # feature 이름 정제
    X = clean_feature_names(X)

    # feature 컬럼명 정제
    X_train, X_test = refine_feature_name(X_train, X_test)

    # box-cox 변환
    y_train_boxcox, y_test_boxcox, lambda_ = box_cox(y_train, y_test)

    # 모델 생성 및 훈련
    best_lgbm_regression = train_model(X_train, y_train_boxcox)

    # 모델 예측
    y_pred_lgbm_grid = eveluation(best_lgbm_regression, X_test, lambda_)


    # ----------------------------- # --------------(데이터 테이블 만들기)--------------- # ----------------------------- #
    # base_data 준비
    base_data = get_base_data()

    X_table, y_table_boxcox, lambda_ = get_y_table_boxcox(base_data)

    # 테이블 데이터 예측
    y_ForTable_pred_lgbm_grid = pred_for_table(best_lgbm_regression, X_table, lambda_)

    # 전처리2
    base_data = get_base_data_2(base_data, y_ForTable_pred_lgbm_grid)

    # 최종 테이블
    expected_income_base_df, district_point_df = get_groupby_base_data(base_data)


    # ----------------------------- # --------------(메인 사이드바 설정)--------------- # ----------------------------- #

    # 시각화 유형 선택
    viz_cat_option = st.sidebar.radio('유형을 선택하세요', ['선택 상권 예상 매출', '예상 매출 종합', '예상 매출 상권 추천'])

    # ----------------------------- # --------------(선택 상권 예상 매출)--------------- # ----------------------------- #
    if viz_cat_option == '선택 상권 예상 매출':

        # ----------------------------- # --------------(관련 사이드바 설정)--------------- # ----------------------------- #

        # 셀렉트 박스-1 : 행정동
        select_district = st.sidebar.selectbox(
        "행정동 선택",
        ('삼성1동', '삼성2동', '개포2동', '개포4동', '역삼1동', '역삼2동', '논현1동', '논현2동',
         '압구정동', '도곡1동', '도곡2동', '청담동', '신사동', '수서동', '대치1동', '대치2동', '대치4동',
         '세곡동', '일원1동')
        )

        if select_district == '삼성1동':
            sub_options = ['강남 마이스 관광특구', '봉은사역', '코엑스', '봉은사역 4번']
        elif select_district == '삼성2동':
            sub_options = ['선정릉역', '포스코사거리', '삼성중앙역']
        elif select_district == '개포2동':
            sub_options = ['강남개포시장', '개포고등학교']
        elif select_district == '개포4동':
            sub_options = ['국악고교사거리', '논현로18길', '포이초등학교(개포목련어린이공원)']
        elif select_district == '역삼1동':
            sub_options = ['구역삼세무서', '역삼역', '뱅뱅사거리', '르네상스호텔사거리', '도곡1동', '경복아파트교차로', '역삼역 8번', '언주역 6번', '선정릉역 4번']
        elif select_district == '역삼2동':
            sub_options = ['개나리아파트', '강남세브란스병원미래의학연구센터']
        elif select_district == '논현1동':
            sub_options = ['학동역', '신논현역', '논현역', '논현초등학교', '논현목련공원']
        elif select_district == '논현2동':
            sub_options = ['서울세관', '언주역(차병원)', '강남구청역', '언주역 3번', '언북중학교']
        elif select_district == '압구정동':
            sub_options = ['성수대교남단', '도산공원교차로', '강남을지병원', '압구정로데오역(압구정로데오)']
        elif select_district == '도곡1동':
            sub_options = ['매봉역 1번']
        elif select_district == '도곡2동':
            sub_options = ['매봉역']
        elif select_district == '청담동':
            sub_options = ['학동사거리', '경기고교사거리(경기고교앞사거리)', '영동대교남단교차로', '강남구청(청담역_8번, 강남세무서)',
                           '청담사거리(청담동명품거리)', '언북초등학교']
        elif select_district == '신사동':
            sub_options = ['압구정역', '가로수길', '한남IC']
        elif select_district == '수서동':
            sub_options = ['수서역']
        elif select_district == '대치1동':
            sub_options = ['대치역']
        elif select_district == '대치2동':
            sub_options = ['휘문고교사거리', '삼성역']
        elif select_district == '대치4동':
            sub_options = ['은마아파트', '대치사거리', '한티역', '도성초등학교', '선릉역', '도곡초등학교', '대치동아우편취급국']
        elif select_district == '세곡동':
            sub_options = ['윗방죽마을공원']
        elif select_district == '일원1동':
            sub_options = ['대청초등학교']

        # 셀렉트 박스-2 : 상권
        select_sub_district = st.sidebar.selectbox("상권 선택", sub_options)

        if select_district == '삼성1동':
            if select_sub_district == '강남 마이스 관광특구':
                district_code = 3001496
            elif select_sub_district == '봉은사역':
                district_code = 3120221
            elif select_sub_district == '코엑스':
                district_code = 3120218
            elif select_sub_district == '봉은사역 4번':
                district_code = 3110995

        elif select_district == '삼성2동':
            if select_sub_district == '선정릉역':
                district_code = 3120207
            elif select_sub_district == '포스코사거리':
                district_code = 3120215
            elif select_sub_district == '삼성중앙역':
                district_code = 3120218

        elif select_district == '개포2동':
            if select_sub_district == '강남개포시장':
                district_code = 3130310
            elif select_sub_district == '개포고등학교':
                district_code = 3110994

        elif select_district == '개포4동':
            if select_sub_district == '국악고교사거리':
                district_code = 3110981
            elif select_sub_district == '논현로18길':
                district_code = 3110977
            elif select_sub_district == '포이초등학교(개포목련어린이공원)':
                district_code = 3110984
        elif select_district == '역삼1동':
            if select_sub_district == '구역삼세무서':
                district_code = 3120198
            elif select_sub_district == '역삼역':
                district_code = 3120197
            elif select_sub_district == '뱅뱅사거리':
                district_code = 3120192
            elif select_sub_district == '르네상스호텔사거리':
                district_code = 3120204
            elif select_sub_district == '도곡1동':
                district_code = 3120201
            elif select_sub_district == '경복아파트교차로':
                district_code = 3120199
            elif select_sub_district == '역삼역 8번':
                district_code = 3110967
            elif select_sub_district == '언주역 6번':
                district_code = 3110965
            elif select_sub_district == '선정릉역 4번':
                district_code = 3110971
        
        elif select_district == '역삼2동':
            if select_sub_district == '개나리아파트':
                district_code = 3120206
            elif select_sub_district == '강남세브란스병원미래의학연구센터':
                district_code = 3110972

        elif select_district == '논현1동':
            if select_sub_district == '학동역':
                district_code = 3120191
            elif select_sub_district == '신논현역':
                district_code = 3120187
            elif select_sub_district == '논현역':
                district_code = 3120185
            elif select_sub_district == '논현초등학교':
                district_code = 3110955
            elif select_sub_district == '논현목련공원':
                district_code = 3110952

        elif select_district == '논현2동':
            if select_sub_district == '서울세관':
                district_code = 3120196
            elif select_sub_district == '언주역(차병원)':
                district_code = 3120194
            elif select_sub_district == '강남구청역':
                district_code = 3120203
            elif select_sub_district == '언주역 3번':
                district_code = 3110961
            elif select_sub_district == '언북중학교':
                district_code = 3110957

        elif select_district == '압구정동':
            if select_sub_district == '성수대교남단':
                district_code = 3120195
            elif select_sub_district == '도산공원교차로':
                district_code = 3120193
            elif select_sub_district == '강남을지병원':
                district_code = 3120190
            elif select_sub_district == '압구정로데오역(압구정로데오)':
                district_code = 3120202

        elif select_district == '도곡1동':
            if select_sub_district == '매봉역 1번':
                district_code = 3110975

        elif select_district == '도곡2동':
            if select_sub_district == '매봉역':
                district_code = 3120205

        elif select_district == '청담동':
            if select_sub_district == '학동사거리':
                district_code = 3120200
            elif select_sub_district == '경기고교사거리(경기고교앞사거리)':
                district_code = 3120216
            elif select_sub_district == '영동대교남단교차로':
                district_code = 3120214
            elif select_sub_district == '강남구청(청담역_8번, 강남세무서)':
                district_code = 3120209
            elif select_sub_district == '청담사거리(청담동명품거리)':
                district_code = 3120208
            elif select_sub_district == '언북초등학교':
                district_code = 3110976

        elif select_district == '신사동':
            if select_sub_district == '압구정역':
                district_code = 3120188
            elif select_sub_district == '가로수길':
                district_code = 3120186
            elif select_sub_district == '한남IC':
                district_code = 3110949

        elif select_district == '수서동':
            if select_sub_district == '수서역':
                district_code = 3120224

        elif select_district == '대치1동':
            if select_sub_district == '대치역':
                district_code = 3120220

        elif select_district == '대치2동':
            if select_sub_district == '휘문고교사거리':
                district_code = 3120223
            elif select_sub_district == '삼성역':
                district_code = 3120222

        elif select_district == '대치4동':
            if select_sub_district == '은마아파트':
                district_code = 3120219
            elif select_sub_district == '대치사거리':
                district_code = 3120217
            elif select_sub_district == '한티역':
                district_code = 3120212
            elif select_sub_district == '도성초등학교':
                district_code = 3120211
            elif select_sub_district == '선릉역':
                district_code = 3120210
            elif select_sub_district == '도곡초등학교':
                district_code = 3110992
            elif select_sub_district == '대치동아우편취급국':
                district_code = 3110989

        elif select_district == '세곡동':
            if select_sub_district == '윗방죽마을공원':
                district_code = 3110999

        elif select_district == '일원1동':
            if select_sub_district == '대청초등학교':
                district_code = 3110997

        # 셀렉트 박스-3 : 분기
        quarter_options = {'1분기':1 ,'2분기':2,'3분기':3,'4분기':4}
        select_quarter = st.sidebar.selectbox(
        '분기 선택',
        list(quarter_options.keys())
        )


        # ----------------------------- # --------------(해당 화면 radio 설정)--------------- # ----------------------------- #
        tab1, tab2 = st.tabs(['시간대별 예상매출 비교', '시간대별 예상매출 상권 지도'])

        # 시각화-1-1
        with tab1:
            # 슬라이싱
            sliced_EIBF_for_1_1 = get_sliced_EIBF_for_1_1(quarter_options[select_quarter], district_code, expected_income_base_df)

            # 시각화
            st.write(f'행정동 : {select_district}')
            st.write(f'상   권 : {select_sub_district}')
            st.write(f'{quarter_options[select_quarter]}분기 시간대별 예상매출 비교 표 : (단위:백만원)')
            st.dataframe(sliced_EIBF_for_1_1.drop(columns=['상권_코드', '행정동_코드', 'center_point', 'latitude', 'longitude']))
            st.plotly_chart(viz_1_1(quarter_options[select_quarter], sliced_EIBF_for_1_1))

        # 시각화-1-2
        with tab2:
            # 셀렉트 박스-4 : 시간대
            time_options = {'00시~06시':'00~06', '06시~11시':'06~11', '11시~14시':'11~14', '14시~17시':'14~17', '17시~21시':'17~21', '21시~24시':'21~24'}
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                select_time = st.selectbox(
                '시간대 선택',
                list(time_options.keys())
                )

            # 슬라이싱
            sliced_EIBF_for_1_2 = get_sliced_EIBF_for_1_2(quarter_options[select_quarter], time_options[select_time], district_code, expected_income_base_df)

            # 시각화
            st.write(f'{quarter_options[select_quarter]}분기, {select_time}')
            st.write(f'{select_district} [{select_sub_district}] 예상매출')
            viz_1_2(sliced_EIBF_for_1_2)

    # ----------------------------- # --------------(예상 매출 종합)--------------- # ----------------------------- #

    elif viz_cat_option == '예상 매출 종합':

        tab3, tab4, tab5 = st.tabs(['분기별, 시간대별 예상매출 종합', '예상 연 매출 종합', '예상 월 매출 종합'])

        # 슬라이싱 for 시각화-3
        sliced_EIBF_for_3 = get_sliced_EIBF_for_3(expected_income_base_df, district_point_df)
        # 슬라이싱 for 시각화-4
        sliced_EIBF_for_4 = get_sliced_EIBF_for_4(sliced_EIBF_for_3)
        
        # 시각화-2
        with tab3:

            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                # 셀렉트 박스-1 : 분기
                quarter_options = {'1분기':1 ,'2분기':2,'3분기':3,'4분기':4}
                select_quarter_a = st.selectbox(
                '분기 선택',
                list(quarter_options.keys()),
                key='select_quarter_a'
                )

            with col6:
                # 셀렉트 박스-2 : 시간대
                time_options = {'00시~06시':'00~06', '06시~11시':'06~11', '11시~14시':'11~14', '14시~17시':'14~17', '17시~21시':'17~21', '21시~24시':'21~24'}
                select_time_a = st.selectbox(
                '시간대 선택',
                list(time_options.keys()),
                key='select_time_a'
                )

            # 슬라이싱
            sliced_EIBF_for_2 = get_sliced_EIBF_for_2(quarter_options[select_quarter_a], time_options[select_time_a], expected_income_base_df)
            st.write(f'{quarter_options[select_quarter_a]}분기 {select_time_a} 예상매출 종합 : (단위:백만원)')

            # 시각화
            viz_2(sliced_EIBF_for_2)

        # 시각화-3
        with tab4:
            st.write('예상 연 매출 종합')

            # 시각화
            viz_3(sliced_EIBF_for_3)
        
        # 시각화-4
        with tab5:
            st.write('예상 월 매출 종합')

            # 시각화
            viz_4(sliced_EIBF_for_4)

    elif viz_cat_option == '예상 매출 상권 추천':
        
        tab6, tab7 = st.tabs(['예상 매출 상권 추천', '예상 순 수익'])

        with tab6:
            # 슬라이더
            wish_income_max = st.slider('시간대 희망매출 최대값(단위:백만)', 0, 3000, 3000)
            wish_income_min = st.slider('시간대 희망매출 최소값(단위:백만)', 0, 3000, 0)

            # 슬라이싱에 필요함-1
            sliced_EIBF_for_3 = get_sliced_EIBF_for_3(expected_income_base_df, district_point_df)
            # 슬라이싱에 필요함-2
            sliced_EIBF_for_4 = get_sliced_EIBF_for_4(sliced_EIBF_for_3)
            # 슬라이싱
            sliced_EIBF_for_5 = get_sliced_EIBF_for_5(wish_income_min, wish_income_max, sliced_EIBF_for_4)

            # 시각화
            viz_5(sliced_EIBF_for_5)

        with tab7:
            None