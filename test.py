import streamlit as st
from streamlit_option_menu import option_menu
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import time
import os

# 가져오기
from expectation import expectation_content


fpath = os.path.join(os.getcwd(), "font/NanumGothic-Regular.ttf")
prop = fm.FontProperties(fname=fpath)

def main():
    st.set_page_config(page_title='강남구 편의점 매출 예측', page_icon="🏪", layout="wide")

    with st.sidebar:
        st.title('메뉴')
        menu = option_menu("메뉴를 선택하세요:", ['홈', '강남구 편의점 분포 현황', '강남구 편의점 매출 현황', '매출 현황 순위', '매출 예측 모델링'],
                   icons=['house', 'map', 'graph-up-arrow', 'cash-coin', 'cpu-fill'], menu_icon="cast", default_index=0)
    
    # CSV 파일 불러오기
    file_path = 'data2/final_reordered.csv'
    df = pd.read_csv(file_path)

    with st.spinner('로딩 중...'):
        time.sleep(0.1)  # Simulating loading time

        if menu == '홈':
            st.markdown("<h1 style='text-align: center;'>강남구 편의점 매출 예측 🏪</h1>", unsafe_allow_html=True)
            st.image('편의점 사진.jpg', use_column_width=True)
            st.image('홈 화면.png', use_column_width=True)

        elif menu == '강남구 편의점 분포 현황': 
            st.markdown("<h1 style='text-align:center;'>강남구 편의점 분포 현황 🗺️ <span style='font-size:smaller;'>(2021년 1분기 ~ 2023년 3분기)</span></h1>", unsafe_allow_html=True)
            st.write('궁금한 상권을 선택하세요 👀')

            # 기존 데이터 프레임과 상권 좌표 정보가 병합된 파일 경로
            merged_file_path = 'data2/map_data.csv'

            # 병합된 데이터 프레임 불러오기
            merged_df = pd.read_csv(merged_file_path)

            # 강남구 중심 좌표
            map_center = [37.5172, 127.0473]

            # Folium 맵 생성
            m = folium.Map(location=map_center, zoom_start=13)

            # MarkerCluster 레이어 생성
            marker_cluster = MarkerCluster().add_to(m)

            # 각 점에 대한 정보를 Folium으로 추가
            for idx, row in merged_df.iterrows():
                avg_sales = row['시간대_매출금액_평균'] / 1_000_000  # 백만원 단위로 변환
                popup_text = f"상권명: {row['상권_코드_명']}, 행정동: {row['행정동_코드_명']}, 시간대_매출금액_평균: {avg_sales:.2f} 백만원"
                folium.Marker([row['latitude'], row['longitude']], popup=popup_text).add_to(marker_cluster)

            # Streamlit에 Folium 맵 표시
            folium_static(m)

        elif menu == '강남구 편의점 매출 현황':
            st.markdown("<h1 style='text-align: center;'>강남구 편의점 매출 현황 📊</h1>", unsafe_allow_html=True)

            # 행정동 코드명 가져오기
            dong_names = df['행정동_코드_명'].unique()
            selected_dong = st.selectbox("행정동 코드명 선택:", dong_names)

            st.write(f"선택된 행정동 코드명: {selected_dong}")

            # 선택된 행정동에 해당하는 데이터 필터링
            selected_dong_data = df[df['행정동_코드_명'] == selected_dong]

            # 행정동 시간대별 평균 매출 시각화
            with st.expander(f"{selected_dong}의 시간대별 평균 매출", expanded=True):
                if not selected_dong_data.empty:
                    fig, ax = plt.subplots()
                    selected_dong_data.groupby('시간대')['시간대_매출금액'].mean().plot(kind='bar', ax=ax, color='skyblue')
                    for i, v in enumerate(selected_dong_data.groupby('시간대')['시간대_매출금액'].mean()):
                        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
                    plt.xticks(fontproperties=prop)  # 폰트 설정
                    plt.yticks(fontproperties=prop)  # 폰트 설정
                    plt.xlabel('시간대', fontsize=12, fontproperties=prop)  # 폰트 설정
                    plt.ylabel('평균 매출금액', fontsize=12, fontproperties=prop)  # 폰트 설정
                    plt.title(f"{selected_dong}의 시간대별 평균 매출", fontsize=14, fontproperties=prop)  # 폰트 설정

                    # y축의 단위 설정
                    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}억".format(x/1e8)))

                    st.pyplot(fig)
                else:
                    st.write("선택된 행정동에 대한 데이터가 없습니다.")

            # 선택된 행정동에 따른 상권 코드명 선택
            filtered_df = df[df['행정동_코드_명'] == selected_dong]
            biz_area_names = filtered_df['상권_코드_명'].unique()
            selected_biz_area = st.selectbox(f"{selected_dong}에 대한 상권 코드명 선택:", biz_area_names)

            # 선택된 상권 코드명에 대한 데이터 필터링
            selected_biz_area_data = df[df['상권_코드_명'] == selected_biz_area]

            # 선택된 상권 코드명에 대한 시간대별 평균 매출 시각화
            with st.expander(f"{selected_biz_area} 상권의 시간대별 평균 매출", expanded=True):
                if not selected_biz_area_data.empty:
                    fig, ax = plt.subplots()
                    selected_biz_area_data.groupby('시간대')['시간대_매출금액'].mean().plot(kind='bar', ax=ax, color='skyblue')
                    for i, v in enumerate(selected_biz_area_data.groupby('시간대')['시간대_매출금액'].mean()):
                        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
                    plt.xticks(fontproperties=prop)  # 폰트 설정
                    plt.yticks(fontproperties=prop)  # 폰트 설정
                    plt.xlabel('시간대', fontsize=12, fontproperties=prop)  # 폰트 설정
                    plt.ylabel('평균 매출금액', fontsize=12, fontproperties=prop)  # 폰트 설정
                    plt.title(f"{selected_biz_area} 상권의 시간대별 평균 매출", fontsize=14, fontproperties=prop)  # 폰트 설정

                    # y축의 단위 설정
                    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}억".format(x/1e8)))

                    st.pyplot(fig)
                else:
                    st.write("선택된 상권에 대한 데이터가 없습니다.")

        elif menu == '매출 현황 순위':
            st.markdown("<h1 style='text-align: center;'>매출 현황 순위 💰</h1>", unsafe_allow_html=True)

            # 각 시간대별 데이터 프레임 생성
            hourly_data = {
                "00:00 ~ 06:00": df[df['시간대'] == '00~06'],
                "06:00 ~ 11:00": df[df['시간대'] == '06~11'],
                "11:00 ~ 14:00": df[df['시간대'] == '11~14'],
                "14:00 ~ 17:00": df[df['시간대'] == '14~17'],
                "17:00 ~ 21:00": df[df['시간대'] == '17~21'],
                "21:00 ~ 24:00": df[df['시간대'] == '21~24']
            }

            # 시간대 선택하는 버튼 생성
            selected_time_range = st.radio("시간대를 선택하세요:",
                                        ['00:00 ~ 06:00', '06:00 ~ 11:00', '11:00 ~ 14:00', '14:00 ~ 17:00', '17:00 ~ 21:00', '21:00 ~ 24:00'])

            # 선택된 시간대에 해당하는 데이터 필터링
            selected_data = hourly_data[selected_time_range]

            # 선택된 시간대에 대한 상권 TOP5 및 시각화 그래프 출력
            if not selected_data.empty:
                top5_by_hour = selected_data.groupby('상권_코드_명')['시간대_매출금액'].mean().nlargest(5)
                

                # 그래프 생성
                fig, ax = plt.subplots()
                top5_by_hour.plot(kind='bar', ax=ax, color='skyblue')
                plt.xlabel("상권", fontsize=12, fontproperties=prop)  # 폰트 설정
                plt.ylabel("평균 매출금액", fontsize=12, fontproperties=prop)  # 폰트 설정
                plt.title(f"{selected_time_range} 시간대 매출이 가장 높은 상권 TOP5", fontsize=14, fontproperties=prop)  # 폰트 설정
                plt.xticks(fontproperties=prop)  # 폰트 설정
                plt.yticks(fontproperties=prop)  # 폰트 설정

                for i, v in enumerate(top5_by_hour):
                    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

                # y축의 단위 설정
                plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}억".format(x/1e8)))

                st.pyplot(fig)
            else:
                st.write("데이터가 없습니다.")

        elif menu == '매출 예측 모델링':
            expectation_content()

if __name__ == "__main__":
    main()