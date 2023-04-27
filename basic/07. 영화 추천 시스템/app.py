import pickle
import streamlit as st
from tmdbv3api import Movie, TMDb

movie = Movie()
tmdb = TMDb()
tmdb.api_key = '2a413d04952ff59f7802b20389da9d6b'
tmdb.language = 'ko-KR' # 한국어로 변환

# 영화 제목 입력하면 코사인 유사도를 통해 가장 유사도 높은 TOP 10 영화 목록 출력
def get_recommendations(title):
    # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
    idx = movies[movies['title'] == title].index[0] # index 값 배열로 넘어오므로 [0] 붙이기
    
    # 코사인 유사도 매트릭스(cosine_sim)에서 idx에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
    sim_scores = list(enumerate(cosine_sim[idx])) # 리스트 형태 아니어도 상관없음

    # 코사인 유사도 기준으로 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
    sim_scores = sim_scores[1:11]

    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indices = [i[0] for i in sim_scores]

    # 인덱스 정보를 통해 이미지, 영화 제목 추출
    images = []
    titles = []

    for i in movie_indices:
        id = movies['id'].iloc[i]
        details = movie.details(id)

        # details['poster_path'] 없을 수도 있으므로 그것을 고려
        image_path = details['poster_path']
        if image_path:
            image_path = 'https://image.tmdb.org/t/p/w500' + image_path # 링크는 홈페이지에 나와 있음
        else:
            image_path = 'no_image.jpg'

        images.append(image_path)
        titles.append(details['title']) # 함수 호출 시 title 받아 왔지만, 나중에 한국어로 전환하기 위해 다시 받아 옴

    return images, titles

movies = pickle.load(open('movies.pickle', 'rb')) # 'rb' : 읽기 모드
cosine_sim = pickle.load(open('cosine_sim.pickle', 'rb'))

st.set_page_config(layout='wide') # 이렇게 하지 않으면 화면 작게 나옴
st.header('Baeflix')

# 영화 목록을 콤보 박스 형태로 출력
movie_list = movies['title'].values
title = st.selectbox('Choose a movie you like', movie_list) # 콤보 박스 형태

# 버튼 생성
if st.button('Recommend'):

    with st.spinner('Please wait...'): # progress bar 생성
        images, titles = get_recommendations(title)

        # 웹페이지에 표현
        idx = 0
        for i in range(0, 2):
            cols = st.columns(5) # 5개의 컬럼 만듦
            for col in cols:
                col.image(images[idx])
                col.write(titles[idx]) # 제목 입력
                idx += 1