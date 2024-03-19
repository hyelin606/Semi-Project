import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

def main():
    st.title('강남구 편의점 분포 현황 🗺️')

    # 기존 데이터 프레임과 상권 좌표 정보가 병합된 파일 경로
    merged_file_path = 'data/map_data.csv'

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
        popup_text = f"상권명: {row['상권_코드_명']}, 행정동: {row['행정동_코드_명']}, 시간대_매출금액_평균: {row['시간대_매출금액_평균']}"
        folium.Marker([row['latitude'], row['longitude']], popup=popup_text).add_to(marker_cluster)

    # Streamlit에 Folium 맵 표시
    folium_static(m)

if __name__ == "__main__":
    main()