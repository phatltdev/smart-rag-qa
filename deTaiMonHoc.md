### MÔN HỌC: ĐỒ ÁN HỌC PHẦN XỬ LÝ NGÔN NGỮ TỰ NHIÊN

## Đề tài: Hệ thống Hỏi - Đáp thông minh dựa trên kỹ thuật RAG (Retrieval-Augmented Generation)


## Yêu cầu:

- 01 file Word báo cáo: Mô tả kiến trúc hệ thống, lý thuyết nền tảng (Transformer, LLMs, RAG, BERT, v.v.), các kỹ thuật tiền xử lý dữ liệu tiếng Việt, và đánh giá chi tiết kết quả thực nghiệm.

- 02 tập tin mã nguồn: Bao gồm 01 tập tin mã nguồn huấn luyện/fined-tune mô hình (khuyến khích viết trên Jupyter Notebook/Google Colab) và 01 tập tin mã nguồn ứng dụng Web (được đóng gói dạng Zip hoặc liên kết Git).

## Yêu cầu kỹ thuật hệ thống: 
- Các đề tài triển khai ứng dụng Web cần có giao diện trực quan (Sử dụng Streamlit, Gradio, FastAPI + React/Vue). Dữ liệu cấu trúc/phi cấu trúc phải được lưu trữ bằng các hệ quản trị CSDL phù hợp (MongoDB, PostgreSQL, hoặc Vector Database như ChromaDB, Milvus, Pinecone).

## Mô tả kỹ thuật:
- Sử dụng 1 trong các bộ dữ liệu tiếng Việt hoặc dữ liệu quy chế học vụ của trường.
https://github.com/ntphuc149/ViLegalQA 
https://www.kaggle.com/datasets/hariwh0/zaloai2021-legal-text-retrieval
- Sử dụng thư viện tách từ tiếng Việt phù hợp (underthesea, pyvi) để tiền xử lý văn bản.
- Sử dụng mô hình Sentence-BERT (như vietnamese-bi-encoder) để chuyển đổi các đoạn văn bản thành Vector Embeddings.
- Lưu trữ các vector này vào một cơ sở dữ liệu vector (Vector Database) như ChromaDB hoặc Milvus.
- Xây dựng hàm tìm kiếm ngữ cảnh tương đồng sử dụng độ đo Cosine.
- Sử dụng một mô hình LLM mã nguồn mở (như PhoGPT, Vistral, hoặc Llama-3-8B-Instruct) kết hợp với Prompt Engineering để tổng hợp câu trả lời từ ngữ cảnh tìm được.