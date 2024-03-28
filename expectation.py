# -*- coding:utf-8 -*-

# 경고 라이브러리
import warnings

# re
import re

# 기본 라이브러리
import pandas as pd
import numpy as np

# 모델 불러오기 라이브러리
import joblib

# GeoPandas 관련
# import geopandas as gpd

# 데이터 분할용
from sklearn.model_selection import train_test_split

# 머신 평가용
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import r2_score

# 스케일링 관련
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler

# 모델
# from sklearn.linear_model import LinearRegression
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


# 모델 불러오기
def load_model():
    loaded_model = joblib.load('data2/best_lgbm_regression.pkl')
    return loaded_model


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


# ----------------------------- # --------------(실 수익 관련)--------------- # ----------------------------- #
    
# 임대료 정보 불러오기
@st.cache_data
def load_rent_df():
    df = pd.read_csv('data2/merged_rent_sales_df.csv')
    return df

# 지출사항-1 : 임대료, 수수료, 마진율
def get_expendi1(margin_rate, franchise_fee, area_size, rent_type, merged_rent_sales_df):
    # 마진율 계산 : 마진 계산을 먼저 하고, 그 수익금 중에서 수수료(보통 30%)를 프랜차이즈가 떼어간다.
    # margin_rate : (단위 : %)
    merged_rent_sales_df['마진 수익'] = merged_rent_sales_df['예상_월매출']*(margin_rate/100)

    # 프렌차이즈 수수료 계산
    # franchise_fee : (단위 : %)
    merged_rent_sales_df['프랜차이즈_수수료'] = merged_rent_sales_df['마진 수익']*(franchise_fee/100)

    # 임대료 계산
    # 넓이 정보
    # area_size :(단위 : 제곱미터)

    # 임대 유형 : '소규모 상가 임대료', '지하1층 임대료', '1층 임대료', '2층 임대료', ...
    # rent_type = '소규모 상가', '지하1층', '1층', '2층', ...

    merged_rent_sales_df['실 임대료'] = area_size * merged_rent_sales_df[rent_type + ' 임대료'] / (1e+3) # 백만단위로 변환

    return merged_rent_sales_df




# ----------------------------- # --------------(이하 main)--------------- # ----------------------------- #
    

def expectation_content():
    st.markdown("<h1 style='text-align:center;'>강남구 편의점 예상매출 종합 🧠</h1>", unsafe_allow_html=True)
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

    # 훈련된 모델 불러오기
    best_lgbm_regression = load_model()

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
            st.plotly_chart(viz_1_1(quarter_options[select_quarter], sliced_EIBF_for_1_1))
            with st.expander('표 확인하기'):
                st.dataframe(sliced_EIBF_for_1_1.drop(columns=['상권_코드', '행정동_코드', 'center_point', 'latitude', 'longitude']))

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
            wish_income_max = st.slider('월 희망매출 최대값(단위:백만)', 0, 3000, 3000)
            wish_income_min = st.slider('월 희망매출 최소값(단위:백만)', 0, 3000, 0)

            # 슬라이싱에 필요함-1
            sliced_EIBF_for_3 = get_sliced_EIBF_for_3(expected_income_base_df, district_point_df)
            # 슬라이싱에 필요함-2
            sliced_EIBF_for_4 = get_sliced_EIBF_for_4(sliced_EIBF_for_3)
            # 슬라이싱
            sliced_EIBF_for_5 = get_sliced_EIBF_for_5(wish_income_min, wish_income_max, sliced_EIBF_for_4)

            # 시각화
            viz_5(sliced_EIBF_for_5)

        with tab7:
            # 임대료 정보 불러오기
            rent_df = load_rent_df()

            # 실 임대료와 마진 계산
            # 입력값 설정
            col9, col10, = st.columns(2)
            with col9:

                # 행정동 셀렉트박스
                select_district_a = st.selectbox(
                "행정동 선택",
                ('삼성1동', '삼성2동', '개포2동', '개포4동', '역삼1동', '역삼2동', '논현1동', '논현2동',
                 '압구정동', '도곡1동', '도곡2동', '청담동', '신사동', '수서동', '대치1동', '대치2동', '대치4동',
                 '세곡동', '일원1동'),
                 key='select_district_a'
                )

                if select_district_a == '삼성1동':
                    sub_options_a = ['강남 마이스 관광특구', '봉은사역', '코엑스', '봉은사역 4번']
                elif select_district_a == '삼성2동':
                    sub_options_a = ['선정릉역', '포스코사거리', '삼성중앙역']
                elif select_district_a == '개포2동':
                    sub_options_a = ['강남개포시장', '개포고등학교']
                elif select_district_a == '개포4동':
                    sub_options_a = ['국악고교사거리', '논현로18길', '포이초등학교(개포목련어린이공원)']
                elif select_district_a == '역삼1동':
                    sub_options_a = ['구역삼세무서', '역삼역', '뱅뱅사거리', '르네상스호텔사거리', '도곡1동', '경복아파트교차로', '역삼역 8번', '언주역 6번', '선정릉역 4번']
                elif select_district_a == '역삼2동':
                    sub_options_a = ['개나리아파트', '강남세브란스병원미래의학연구센터']
                elif select_district_a == '논현1동':
                    sub_options_a = ['학동역', '신논현역', '논현역', '논현초등학교', '논현목련공원']
                elif select_district_a == '논현2동':
                    sub_options_a = ['서울세관', '언주역(차병원)', '강남구청역', '언주역 3번', '언북중학교']
                elif select_district_a == '압구정동':
                    sub_options_a = ['성수대교남단', '도산공원교차로', '강남을지병원', '압구정로데오역(압구정로데오)']
                elif select_district_a == '도곡1동':
                    sub_options_a = ['매봉역 1번']
                elif select_district_a == '도곡2동':
                    sub_options_a = ['매봉역']
                elif select_district_a == '청담동':
                    sub_options_a = ['학동사거리', '경기고교사거리(경기고교앞사거리)', '영동대교남단교차로', '강남구청(청담역_8번, 강남세무서)',
                                   '청담사거리(청담동명품거리)', '언북초등학교']
                elif select_district_a == '신사동':
                    sub_options_a = ['압구정역', '가로수길', '한남IC']
                elif select_district_a == '수서동':
                    sub_options_a = ['수서역']
                elif select_district_a == '대치1동':
                    sub_options_a = ['대치역']
                elif select_district_a == '대치2동':
                    sub_options_a = ['휘문고교사거리', '삼성역']
                elif select_district_a == '대치4동':
                    sub_options_a = ['은마아파트', '대치사거리', '한티역', '도성초등학교', '선릉역', '도곡초등학교', '대치동아우편취급국']
                elif select_district_a == '세곡동':
                    sub_options_a = ['윗방죽마을공원']
                elif select_district_a == '일원1동':
                    sub_options_a = ['대청초등학교']

                margin_rate = st.number_input('마진율을 입력하세요(%)')
                area_size = st.number_input('매장 넓이를 입력하세요(제곱미터)')


            with col10:

                # 상권 셀렉트박스
                select_sub_district_a = st.selectbox("상권 선택", sub_options_a, key='elect_sub_district_a')

                if select_district_a == '삼성1동':
                    if select_sub_district_a == '강남 마이스 관광특구':
                        district_code_a = 3001496
                    elif select_sub_district_a == '봉은사역':
                        district_code_a = 3120221
                    elif select_sub_district_a == '코엑스':
                        district_code_a = 3120218
                    elif select_sub_district_a == '봉은사역 4번':
                        district_code_a = 3110995

                elif select_district_a == '삼성2동':
                    if select_sub_district_a == '선정릉역':
                        district_code_a = 3120207
                    elif select_sub_district_a == '포스코사거리':
                        district_code_a = 3120215
                    elif select_sub_district_a == '삼성중앙역':
                        district_code_a = 3120218

                elif select_district_a == '개포2동':
                    if select_sub_district_a == '강남개포시장':
                        district_code_a = 3130310
                    elif select_sub_district_a == '개포고등학교':
                        district_code_a = 3110994

                elif select_district_a == '개포4동':
                    if select_sub_district_a == '국악고교사거리':
                        district_code_a = 3110981
                    elif select_sub_district_a == '논현로18길':
                        district_code_a = 3110977
                    elif select_sub_district_a == '포이초등학교(개포목련어린이공원)':
                        district_code_a = 3110984
                elif select_district_a == '역삼1동':
                    if select_sub_district_a == '구역삼세무서':
                        district_code_a = 3120198
                    elif select_sub_district_a == '역삼역':
                        district_code_a = 3120197
                    elif select_sub_district_a == '뱅뱅사거리':
                        district_code_a = 3120192
                    elif select_sub_district_a == '르네상스호텔사거리':
                        district_code_a = 3120204
                    elif select_sub_district_a == '도곡1동':
                        district_code_a = 3120201
                    elif select_sub_district_a == '경복아파트교차로':
                        district_code_a = 3120199
                    elif select_sub_district_a == '역삼역 8번':
                        district_code_a = 3110967
                    elif select_sub_district_a == '언주역 6번':
                        district_code_a = 3110965
                    elif select_sub_district_a == '선정릉역 4번':
                        district_code_a = 3110971
        
                elif select_district_a == '역삼2동':
                    if select_sub_district_a == '개나리아파트':
                        district_code_a = 3120206
                    elif select_sub_district_a == '강남세브란스병원미래의학연구센터':
                        district_code_a = 3110972

                elif select_district_a == '논현1동':
                    if select_sub_district_a == '학동역':
                        district_code_a = 3120191
                    elif select_sub_district_a == '신논현역':
                        district_code_a = 3120187
                    elif select_sub_district_a == '논현역':
                        district_code_a = 3120185
                    elif select_sub_district_a == '논현초등학교':
                        district_code_a = 3110955
                    elif select_sub_district_a == '논현목련공원':
                        district_code_a = 3110952

                elif select_district_a == '논현2동':
                    if select_sub_district_a == '서울세관':
                        district_code_a = 3120196
                    elif select_sub_district_a == '언주역(차병원)':
                        district_code_a = 3120194
                    elif select_sub_district_a == '강남구청역':
                        district_code_a = 3120203
                    elif select_sub_district_a == '언주역 3번':
                        district_code_a = 3110961
                    elif select_sub_district_a == '언북중학교':
                        district_code_a = 3110957

                elif select_district_a == '압구정동':
                    if select_sub_district_a == '성수대교남단':
                        district_code_a = 3120195
                    elif select_sub_district_a == '도산공원교차로':
                        district_code_a = 3120193
                    elif select_sub_district_a == '강남을지병원':
                        district_code_a = 3120190
                    elif select_sub_district_a == '압구정로데오역(압구정로데오)':
                        district_code_a = 3120202

                elif select_district_a == '도곡1동':
                    if select_sub_district_a == '매봉역 1번':
                        district_code_a = 3110975

                elif select_district_a == '도곡2동':
                    if select_sub_district_a == '매봉역':
                        district_code_a = 3120205

                elif select_district_a == '청담동':
                    if select_sub_district_a == '학동사거리':
                        district_code_a = 3120200
                    elif select_sub_district_a == '경기고교사거리(경기고교앞사거리)':
                        district_code_a = 3120216
                    elif select_sub_district_a == '영동대교남단교차로':
                        district_code_a = 3120214
                    elif select_sub_district_a == '강남구청(청담역_8번, 강남세무서)':
                        district_code_a = 3120209
                    elif select_sub_district_a == '청담사거리(청담동명품거리)':
                        district_code_a = 3120208
                    elif select_sub_district_a == '언북초등학교':
                        district_code_a = 3110976

                elif select_district_a == '신사동':
                    if select_sub_district_a == '압구정역':
                        district_code_a = 3120188
                    elif select_sub_district_a == '가로수길':
                        district_code_a = 3120186
                    elif select_sub_district_a == '한남IC':
                        district_code_a = 3110949

                elif select_district_a == '수서동':
                    if select_sub_district_a == '수서역':
                        district_code_a = 3120224

                elif select_district_a == '대치1동':
                    if select_sub_district_a == '대치역':
                        district_code_a = 3120220

                elif select_district_a == '대치2동':
                    if select_sub_district_a == '휘문고교사거리':
                        district_code_a = 3120223
                    elif select_sub_district_a == '삼성역':
                        district_code_a = 3120222

                elif select_district_a == '대치4동':
                    if select_sub_district_a == '은마아파트':
                        district_code_a = 3120219
                    elif select_sub_district_a == '대치사거리':
                        district_code_a = 3120217
                    elif select_sub_district_a == '한티역':
                        district_code_a = 3120212
                    elif select_sub_district_a == '도성초등학교':
                        district_code_a = 3120211
                    elif select_sub_district_a == '선릉역':
                        district_code_a = 3120210
                    elif select_sub_district_a == '도곡초등학교':
                        district_code_a = 3110992
                    elif select_sub_district_a == '대치동아우편취급국':
                        district_code_a = 3110989

                elif select_district_a == '세곡동':
                    if select_sub_district_a == '윗방죽마을공원':
                        district_code_a = 3110999

                elif select_district_a == '일원1동':
                    if select_sub_district_a == '대청초등학교':
                        district_code_a = 3110997


                franchise_fee = st.number_input('프랜차이즈 수수료율을 입력하세요(%)')
                rent_type = st.selectbox('임대 유형', ('소규모 상가', '지하1층', '1층', '2층', '3층', '4층', '5층'))


            merged_rent_sales_df = get_expendi1(margin_rate, franchise_fee, area_size, rent_type, rent_df)
            # st.dataframe(merged_rent_sales_df)

            st.write('-'*50)
            st.write('인건비 관련사항 입력')

            # 인건비 지출사항
            col11, col12, col13 = st.columns(3)

            with col11:
                # 셀렉트 박스-1 : 날짜 유형 : '평일', '주말' : [일 수, 타입코드]
                week_dict = {'평일': [5, 1], '주말': [2, 2]}
                week_type = st.selectbox('근무 유형 선택', list(week_dict.keys()))
            
            with col12:
                # 셀렉트 박스-2 : 시간대 : [시간, 타입코드]
                time_dict = {'오픈(09~16)':[7, 1], '저녁(16~23)':[7, 2], '야간(23~09)':[10, 3]}
                time_type = st.selectbox('근무 시간대 선택', list(time_dict.keys()))

            with col13:
                # 시간당 임금
                pay_per_hour = st.number_input('시급을 입력하세요(원)')

                # 월급 계산
                pay_per_month = (week_dict[week_type][0] * 4 * time_dict[time_type][0] * pay_per_hour) / 1e+6


            # 주말알바, 평일알바 수
            init_n_week = 0
            init_n_weekend = 0

            # 시간대 알바 수
            init_n_open = 0
            init_n_day = 0
            init_n_night = 0

            # 세션 스테이트 지정
            if 'n_week' not in st.session_state:
                st.session_state.n_week = init_n_week
            if 'n_weekend' not in st.session_state:
                st.session_state.n_weekend = init_n_weekend
            if 'n_open' not in st.session_state:
                st.session_state.n_open = init_n_open
            if 'n_day' not in st.session_state:
                st.session_state.n_day = init_n_day
            if 'n_night' not in st.session_state:
                st.session_state.n_night = init_n_night


            # 월급 사항을 리스트로 관리
            init_pay_list = []
            # 세션 스테이트 지정
            if 'pay_list' not in st.session_state:
                st.session_state.pay_list = init_pay_list

            # 빈 데이터 프레임
            init_arbeiter_df = pd.DataFrame(columns=['종업원', '근무 유형', '시간대', '월간 급여'])
            # 세션 스테이트 지정
            if 'arbeiter_df' not in st.session_state:
                st.session_state.arbeiter_df = init_arbeiter_df

            init_abc = 0
            # 세션 스테이트 지정
            if 'abc' not in st.session_state:
                st.session_state.abc = init_abc

            init_n_arbeiter_sum = 0
            # 세션 스테이트 지정
            if 'n_arbeiter_sum' not in st.session_state:
                st.session_state.n_arbeiter_sum = init_n_arbeiter_sum


            # '추가' 버튼
            add_button = st.button('추가')
            reset_button = st.button('초기화', type='primary')
            
            if add_button:
                # 알바생 변화량
                if week_dict[week_type][1] == 1:
                    st.session_state.n_week += 1
                if week_dict[week_type][1] == 2:
                    st.session_state.n_weekend += 1
                if time_dict[time_type][1] == 1:
                    st.session_state.n_open += 1
                if time_dict[time_type][1] == 2:
                    st.session_state.n_day += 1
                if time_dict[time_type][1] == 3:
                    st.session_state.n_night += 1

                # 전체 알바 수
                st.session_state.n_arbeiter_sum = st.session_state.n_open + st.session_state.n_day + st.session_state.n_night

                # 데이터 프레임 출력
                new_row = pd.DataFrame({'종업원':[f'종업원{st.session_state.abc+1}'], '근무 유형':[[key for key, value in week_dict.items() if value == week_dict[week_type]][0]], '시간대':[[key for key, value in time_dict.items() if value == time_dict[time_type]][0]], '월간 급여':[pay_per_month]})
                # new_DataFrame = pd.DataFrame([new_row])
                st.session_state.arbeiter_df = pd.concat([st.session_state.arbeiter_df, new_row], ignore_index=True)

                st.session_state.abc = st.session_state.abc+1

                # 버튼을 누를 때 pay_list에 월급이 추가됨
                st.session_state.pay_list.append(pay_per_month)


            # 세션 스테이트 초기화
            if reset_button:
                st.session_state.clear()

                # 세션 스테이트 지정
                if 'arbeiter_df' not in st.session_state:
                    st.session_state.arbeiter_df = init_arbeiter_df

                # 세션 스테이트 지정
                if 'pay_list' not in st.session_state:
                    st.session_state.pay_list = init_pay_list

                # 세션 스테이트 지정
                if 'n_week' not in st.session_state:
                    st.session_state.n_week = init_n_week
                if 'n_weekend' not in st.session_state:
                    st.session_state.n_weekend = init_n_weekend
                if 'n_open' not in st.session_state:
                    st.session_state.n_open = init_n_open
                if 'n_day' not in st.session_state:
                    st.session_state.n_day = init_n_day
                if 'n_night' not in st.session_state:
                    st.session_state.n_night = init_n_night

                # 세션 스테이트 지정
                if 'n_arbeiter_sum' not in st.session_state:
                    st.session_state.n_arbeiter_sum = init_n_arbeiter_sum


            st.write('-'*50)

            expendi = merged_rent_sales_df.loc[merged_rent_sales_df['상권_코드']==district_code_a, ['프랜차이즈_수수료', '실 임대료']].sum().sum()
            expendi_sum = expendi + sum(st.session_state.pay_list)
            # expendi_sum

            # 마진 수익금
            margin_income = merged_rent_sales_df.loc[merged_rent_sales_df['상권_코드']==district_code_a, '마진 수익'].reset_index(drop=True)[0]
            margin_income

            # 실 수익 = 마진 수익금 - 지출금 : 인건비'추가' 버튼을 누른 다음 코드가 한 번 돌고, 다시 인건비 정보를 바꿨을 때, '추가' 버튼을 누르기 전에 여기에 값이 한 번 전달된다. 왜냐? 정보가 변경되었으니까.
            # 다만 '추가'버튼을 누르기 전이라서 인건비 정보가 들어가지는 않았지만, 루프가 한 번 더 돌았기 때문에 이전의 인건비 값이 한 번 더 계산되는 것이다.


            init_true_income = 0
            # 세션 스테이트 설정
            if 'true_income' not in st.session_state:
                st.session_state.true_income = init_true_income


            st.session_state.true_income = round(margin_income - expendi_sum, 3) # margin_income이 아직 정의되지 않았는데 불러와서 문제가 생김.
            
            st.header(f'월간 실 수익금(단위:백만원) : {st.session_state.true_income}')
            st.write('')
            st.write('종업원 정보 (월간 급여 단위 : 백만원)')
            st.dataframe(st.session_state.arbeiter_df)
            
            st.write('종업원 근무 유형별 분류')
            st.write(f'주간 종업원 수 : {st.session_state.n_week}')
            st.write(f'주말 종업원 수 : {st.session_state.n_weekend}')
            st.write('-'*10)
            st.write('종업원 근무 시간대별 분류')
            st.write(f'아침 시간대 종업원 수 : {st.session_state.n_open}')
            st.write(f'저녁 시간대 종업원 수 : {st.session_state.n_day}')
            st.write(f'야간 시간대 종업원 수 : {st.session_state.n_night}')
            st.write('-'*10)
            st.write(f'종업원 수 총 합 : {st.session_state.n_arbeiter_sum}')