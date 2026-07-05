import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Giao diện tiêu đề đơn giản
st.title("📚 Trợ Lý Gợi Ý Trích Dẫn")
st.write("---")

# 2. Kho dữ liệu bài báo mẫu có sẵn (Bạn có thể sửa chữ trong này tùy ý)
data_papers = [
    {"title": "Nghiên cứu về Trí Tuệ Nhân Tạo 2026", "abstract": "Mô hình ngôn ngữ lớn và ứng dụng học máy vào thực tế đời sống xã hội.", "path": "Thư_mục_gốc > AI_Lab > Papers"},
    {"title": "Xử lý ngôn ngữ tự nhiên tiếng Việt", "abstract": "Phương pháp tách từ và phân tích ngữ nghĩa văn bản bằng học sâu Transformer.", "path": "Thư_mục_gốc > NLP_Team > Docs"},
    {"title": "Tổng quan về hệ thống gợi ý trích dẫn", "abstract": "Sử dụng thuật toán TF-IDF và cosine để tìm bài báo khoa học tương đồng văn cảnh.", "path": "Thư_mục_gốc > Luận_văn > Tham_khảo"}
]

# 3. Ô nhập văn bản của bạn
user_query = st.text_area("Nhập đoạn văn hoặc ý tưởng của bạn vào đây:", "Tôi muốn tìm hiểu về mô hình ngôn ngữ lớn transformer")

# 4. Bấm nút để tìm kiếm
if st.button("🚀 Tìm Bài Báo Phù Hợp", type="primary"):
    # Thuật toán backend tính toán độ tương đồng từ vựng
    abstracts = [p["abstract"] for p in data_papers] + [user_query]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(abstracts)
    
    # Tính điểm tương đồng giữa câu bạn nhập (dòng cuối) và các bài báo
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    st.write("### Kết quả gợi ý:")
    for i, score in enumerate(scores):
        if score > 0: # Chỉ hiện bài báo có từ khóa liên quan
            st.markdown(f"**📄 {data_papers[i]['title']}** *(Độ khớp: {score:.1%})*")
            st.caption(f"📂 Đường dẫn lưu trữ: {data_papers[i]['path']}")
            st.info(f"Tóm tắt: {data_papers[i]['abstract']}")
            st.write("---")