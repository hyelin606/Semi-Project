import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
    st.sidebar.title('메뉴')

    menu = st.sidebar.radio('메뉴를 선택하세요:', ['홈', '현재 강남구 편의점 분포', '기존 강남구 편의점 시간대별 매출'])

    if menu == '홈':
        st.title('강남구 편의점 시간대별 매출 예측 서비스')
        st.write('편의점 예비 창업자들을 위한 시간대별 매출 예측 서비스입니다!')
    
    elif menu == '현재 강남구 편의점 분포': 
        st.title('현 강남구 편의점 분포 현황')
        st.write('행정동에 따른 편의점 점포 수 지도 시각화 보여주기')
    

    elif menu == '기존 강남구 편의점 시간대별 매출':
        st.title('기존 강남구 편의점 시간대별 매출')

        # CSV 파일 불러오기
        file_path = 'data/final_reordered.csv'
        df = pd.read_csv(file_path)

        # 행정동 코드명 가져오기
        dong_names = df['행정동_코드_명'].unique()

        # 셀렉트 박스 생성
        selected_dong = st.selectbox("행정동 코드명 선택:", dong_names)

        # 선택된 행정동에 대한 데이터 필터링
        filtered_df = df[df['행정동_코드_명'] == selected_dong]

        # 시간대를 역순으로 정렬
        time_order = ['21~24', '17~21', '14~17', '11~14', '06~11', '00~06']
        filtered_df = filtered_df.iloc[::-1]

        # 시각화
        st.write(f"{selected_dong}의 시간대별 평균 매출금액:")
        fig, ax = plt.subplots()

        ax.bar(filtered_df['시간대'], filtered_df['시간대_매출금액'])

        # Nanum Gothic 폰트를 플롯에 직접 임베드
        font_path = "C:\\multicampus\\Semi-Project\\Nanum_Gothic\\NanumGothic-Regular.ttf"
        prop = FontProperties(fname=font_path)
        ax.set_xlabel('시간대', fontproperties=prop)
        ax.set_ylabel('평균 매출금액', fontproperties=prop)
        ax.set_title(f"{selected_dong}의 시간대별 평균 매출금액", fontproperties=prop)

        st.pyplot(fig)

if __name__ == "__main__":
    main()