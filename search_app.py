"""Streamlit 검색 화면.  실행: streamlit run search_app.py"""
import io

import numpy as np
import streamlit as st
from PIL import Image

from search import FigureSearcher

st.set_page_config(page_title="교재 그림 검색", layout="wide")
st.title("교재 그림 검색")


@st.cache_resource            # DINOv2 + 임베딩은 앱 시작 시 1번만 로드
def load_searcher():
    return FigureSearcher()


searcher = load_searcher()
st.caption(f"검색 대상 PDF: {searcher.pdf_path.name if searcher.pdf_path else '없음'}"
           f" · 저장된 그림 {len(searcher.meta)}개")

with st.sidebar:
    mode = st.radio("검색 모드", ["edge", "raw"],
                    help="edge: 윤곽선 기반(스케치용), raw: 원본 회색조")
    k = st.slider("결과 페이지 수", 1, 5, 3)
    show_page = st.checkbox("PDF 페이지 전체 보기", value=True)

up = st.file_uploader("스케치 또는 그림 이미지 업로드", type=["png", "jpg", "jpeg"])

if up is not None:
    rgb = np.array(Image.open(io.BytesIO(up.getvalue())).convert("RGB"))  # RGB uint8

    col_q, col_r = st.columns([1, 3])
    with col_q:
        st.image(rgb, caption="입력 이미지", use_container_width=True)

    with col_r:
        if st.button("검색", type="primary"):
            with st.spinner("검색 중..."):
                results = searcher.search(rgb, mode=mode, k=k)

            for r in results:
                st.subheader(f"{r['rank']}위 · {r['page']}쪽 (유사도 {r['score']:.3f})")
                st.caption(f"{r['fig_id']} · {r['level']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(r["file"], caption="일치한 그림", use_container_width=True)
                if show_page:
                    page_png = searcher.render_page(r["page"])
                    if page_png:
                        with c2:
                            st.image(page_png, caption=f"PDF {r['page']}쪽",
                                     use_container_width=True)
                st.divider()
