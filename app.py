import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 페이지 설정
st.set_page_config(page_title="한반도 기후변화 RAG 챗봇", page_icon="🌍")

st.title("🌍 한반도 100년의 기후변화 RAG 챗봇")
st.write("강의 자료 기반의 기후변화 문서 데이터를 검색하고 답변을 생성하는 미니 RAG 프로그램입니다.")

# 1. 샘플 문서 데이터 로드 및 벡터 DB 구축 (캐싱 처리)
@st.cache_resource
def load_data_and_model():
    documents = [
        "지난 106년(1912~2017) 동안 우리나라의 연평균기온 변화량은 +0.18℃/10년으로 상승하였습니다.",
        "계절별로는 겨울(+0.25℃/10년)과 봄(+0.24℃/10년)의 기온상승이 가장 뚜렷합니다.",
        "여름은 19일 길어졌고 겨울은 18일 짧아졌으며, 폭염과 호우 등의 발생 빈도가 증가하고 있습니다.",
        "지난 106년 동안 강한 강수는 증가하고 약한 강수는 감소하는 경향을 보였습니다.",
        "자연재해로 인하여 최근 10년(2007~2016) 동안 162명의 인명피해와 약 6조 3천억 원의 재산피해가 발생했습니다."
    ]
    
    # 임베딩 모델 로드
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    doc_embeddings = model.encode(documents, convert_to_numpy=True).astype(np.float32)
    
    # FAISS 벡터 인덱스 구축
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    
    return documents, model, index

documents, model, index = load_data_and_model()

# 2. 사용자 질의 입력 UI
user_query = st.text_input("기후변화에 대해 궁금한 점을 입력하세요:", "기온 변화는 어떻게 되나요?")

if st.button("검색 및 답변 생성"):
    if user_query:
        # 질의 벡터화 및 FAISS 검색 (Top-1)
        query_vec = model.encode([user_query], convert_to_numpy=True).astype(np.float32)
        distances, indices = index.search(query_vec, 1)
        
        best_idx = indices[0][0]
        retrieved_chunk = documents[best_idx]
        distance = distances[0][0]
        
        # 결과 출력
        st.subheader("🔍 RAG 검색 결과 (Retrieval)")
        st.info(f"참고한 문서 조각 (유사도 거리: {distance:.4f}):\n\n> {retrieved_chunk}")
        
        st.subheader("🤖 AI 생성 답변 (Generation)")
        st.success(f"문서 내용을 근거로 답변합니다:\n\n질문하신 내용과 가장 관련된 보고서 데이터에 따르면, {retrieved_chunk} 이것이 주요 특징입니다.")
    else:
        st.warning("질문을 입력해주세요!")


