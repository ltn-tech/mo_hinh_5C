import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==========================================
# KHỞI TẠO PAGE CONFIG (Bắt buộc đầu tiên)
# ==========================================
st.set_page_config(
    page_title="Dự báo Rủi ro Khách hàng (Mô hình 5C)",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# HÀM CACHE DÙNG CHUNG
# ==========================================
@st.cache_data
def load_data(file):
    """Đọc dữ liệu từ file upload"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

# Các biến đầu vào từ notebook
FEATURES = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'NL1', 'NL2', 'NL3', 'NL4', 'DK1',
            'DK2', 'DK3', 'DK4', 'DK5', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'TS1',
            'TS2', 'TS3', 'TS4']
TARGET = 'PD'

# ==========================================
# THÀNH PHẦN 1: SIDEBAR - CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    uploaded_file = st.file_uploader("Tải lên dữ liệu mẫu (CSV/Excel)", type=['csv', 'xlsx'])
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Logistic Regression")
    
    # Tham số chia dữ liệu (lấy random_state=23 từ notebook làm mặc định)
    test_size = st.slider("Tỷ lệ tập kiểm thử (Test Size)", 0.1, 0.5, 0.2, 0.05, help="Tỷ lệ chia dữ liệu cho tập test.")
    random_state_split = st.number_input("Random State (Chia dữ liệu)", value=23, step=1, help="Đảm bảo kết quả chia dữ liệu có thể tái lập giống notebook.")
    
    with st.expander("Tham số nâng cao (Hyperparameters)"):
        c_param = st.number_input("C (Nghịch đảo độ chuẩn hóa)", value=1.0, step=0.1, help="Giá trị C nhỏ chỉ định chuẩn hóa mạnh hơn.")
        max_iter = st.number_input("max_iter (Số vòng lặp tối đa)", value=100, step=10)
        solver = st.selectbox("Thuật toán tối ưu (solver)", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])

    st.divider()
    # NÚT HUẤN LUYỆN
    train_button = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# THÀNH PHẦN 2: HEADER - ĐỊNH HƯỚNG
# ==========================================
st.title("🏦 Đánh giá Rủi ro theo Mô hình 5C")
st.caption("Ứng dụng hỗ trợ dự báo rủi ro khách hàng dựa trên thuật toán Logistic Regression và dữ liệu 5 khía cạnh (Tài chính, Năng lực, Điều kiện, Vốn, Tài sản).")

if uploaded_file is None:
    st.info("👋 Vui lòng tải lên file dữ liệu (ví dụ: `5c.csv`) ở thanh bên trái (Sidebar) để bắt đầu.")
    st.stop()

# Đọc dữ liệu
try:
    df = load_data(uploaded_file)
    st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")
except Exception as e:
    st.error(f"Lỗi khi đọc file: {e}")
    st.stop()

st.divider()

# ==========================================
# XỬ LÝ KHỐI TRAIN (Kích hoạt từ Sidebar)
# ==========================================
if train_button:
    # Kiểm tra cột
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing_cols:
        st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc: {missing_cols}")
    else:
        with st.spinner("Đang huấn luyện mô hình..."):
            # Lấy dữ liệu
            X = df[FEATURES]
            y = df[TARGET]
            
            # Chia tập train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state_split
            )
            
            # Khởi tạo và huấn luyện mô hình
            model = LogisticRegression(C=c_param, max_iter=max_iter, solver=solver)
            model.fit(X_train, y_train)
            
            # Dự báo trên tập test
            yhat_test = model.predict(X_test)
            
            # Lưu vào session state
            st.session_state['model_trained'] = True
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['yhat_test'] = yhat_test
            
            st.toast("✅ Huấn luyện thành công!", icon="🎉")

# ==========================================
# TẠO TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan Dữ liệu", 
    "📈 Trực quan hóa", 
    "🎯 Kiểm định Mô hình", 
    "🔮 Sử dụng Mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TỔNG QUAN DỮ LIỆU
# ------------------------------------------
with tab1:
    st.subheader("Kích thước dữ liệu")
    col1, col2 = st.columns(2)
    col1.metric("Số lượng bản ghi (Dòng)", df.shape[0])
    col2.metric("Số lượng thuộc tính (Cột)", df.shape[1])
    
    st.subheader("Dữ liệu thô (5 dòng đầu)")
    with st.container(height=250):
        st.dataframe(df.head(), use_container_width=True)
        
    st.subheader("Thống kê mô tả (Các biến đưa vào mô hình)")
    valid_cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    st.dataframe(df[valid_cols].describe(), use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------
with tab2:
    st.subheader("Phân phối các biến quan trọng")
    valid_cols = [c for c in [TARGET] + FEATURES if c in df.columns]
    
    selected_vars = st.multiselect(
        "Chọn biến để trực quan hóa:", 
        options=valid_cols, 
        default=[TARGET, FEATURES[0], FEATURES[5], FEATURES[9]] if len(valid_cols) > 4 else valid_cols,
        max_selections=6
    )
    
    if selected_vars:
        cols = st.columns(2)
        for idx, var in enumerate(selected_vars):
            with cols[idx % 2]:
                # Hầu hết dữ liệu 5C là dạng phân loại/ordinal (1-5, 0-1)
                val_counts = df[var].value_counts().reset_index()
                val_counts.columns = [var, 'Số lượng']
                val_counts = val_counts.sort_values(by=var)
                
                fig = px.bar(val_counts, x=var, y='Số lượng', text='Số lượng', 
                             title=f"Phân phối của biến: {var}",
                             category_orders={var: sorted(df[var].unique())})
                # Nếu là biến mục tiêu PD
                if var == TARGET:
                    fig.update_traces(marker_color=['#2ECC71' if x==0 else '#E74C3C' for x in val_counts[var]])
                
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 5: KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------
with tab3:
    if 'model_trained' not in st.session_state:
        st.info("Vui lòng cấu hình và bấm nút **🚀 Huấn luyện mô hình** ở thanh bên trái trước khi xem kết quả.")
    else:
        y_test = st.session_state['y_test']
        yhat_test = st.session_state['yhat_test']
        
        st.subheader("Chỉ tiêu đánh giá trên tập kiểm thử (Test Set)")
        acc = accuracy_score(y_test, yhat_test)
        
        col1, col2 = st.columns(2)
        col1.metric("Độ chính xác (Accuracy)", f"{acc*100:.2f}%")
        
        with col1:
            st.markdown("**Báo cáo chi tiết (Classification Report):**")
            report = classification_report(y_test, yhat_test, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
        with col2:
            st.markdown("**Ma trận nhầm lẫn (Confusion Matrix):**")
            cm = confusion_matrix(y_test, yhat_test)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Dự báo (yhat)", y="Thực tế (y_test)", color="Số lượng"),
                               x=['Không Rủi ro (0)', 'Rủi ro (1)'],
                               y=['Không Rủi ro (0)', 'Rủi ro (1)'])
            st.plotly_chart(fig_cm, use_container_width=True)
            
        st.caption("*Đường chéo chính của ma trận nhầm lẫn thể hiện số lượng dự báo chính xác.*")

# ------------------------------------------
# THÀNH PHẦN 6: SỬ DỤNG MÔ HÌNH
# ------------------------------------------
with tab4:
    if 'model_trained' not in st.session_state:
        st.info("Vui lòng cấu hình và bấm nút **🚀 Huấn luyện mô hình** ở thanh bên trái để sử dụng chức năng này.")
    else:
        model = st.session_state['model']
        
        mode = st.radio("Chọn phương thức dự báo:", ["Nhập tay từng khách hàng", "Dự báo hàng loạt từ File"])
        
        if mode == "Nhập tay từng khách hàng":
            st.markdown("### Nhập thông tin khảo sát 5C")
            with st.form("predict_form"):
                input_data = {}
                
                # Trình bày UI dạng lưới cho đẹp mắt
                cols_ui = st.columns(4)
                for i, feature in enumerate(FEATURES):
                    # Giả định giá trị từ 1 đến 5 theo dạng thang đo likert
                    min_val = int(df[feature].min()) if feature in df.columns else 1
                    max_val = int(df[feature].max()) if feature in df.columns else 5
                    default_val = int(df[feature].median()) if feature in df.columns else 3
                    
                    with cols_ui[i % 4]:
                        input_data[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=default_val, step=1)
                
                submit = st.form_submit_button("🔮 Chấm điểm & Dự báo", type="primary")
                
            if submit:
                # Chuyển input thành DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Dự báo
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                st.divider()
                st.subheader("Kết quả dự báo")
                
                res_col1, res_col2 = st.columns(2)
                if prediction == 0:
                    res_col1.success("✅ Dự báo: KHÔNG CÓ RỦI RO (0)")
                else:
                    res_col1.error("⚠️ Dự báo: CÓ RỦI RO (1)")
                    
                res_col2.metric("Xác suất Không rủi ro (0)", f"{probabilities[0]*100:.2f}%")
                res_col2.metric("Xác suất Có rủi ro (1)", f"{probabilities[1]*100:.2f}%")
                
        else:
            st.markdown("### Tải lên tệp khách hàng mới")
            st.caption(f"Tệp cần có đầy đủ {len(FEATURES)} cột: {', '.join(FEATURES)}")
            
            scoring_file = st.file_uploader("Tải tệp cần dự báo", type=['csv', 'xlsx'], key='scoring')
            
            if scoring_file:
                scoring_df = load_data(scoring_file)
                missing = [col for col in FEATURES if col not in scoring_df.columns]
                
                if missing:
                    st.error(f"Tệp tải lên bị thiếu các cột: {missing}")
                else:
                    scoring_features = scoring_df[FEATURES]
                    preds = model.predict(scoring_features)
                    probs = model.predict_proba(scoring_features)
                    
                    scoring_df['Dự báo (PD)'] = preds
                    scoring_df['Xác suất (0)'] = probs[:, 0].round(4)
                    scoring_df['Xác suất (1)'] = probs[:, 1].round(4)
                    
                    st.success(f"Đã chấm điểm thành công {len(scoring_df)} hồ sơ!")
                    
                    with st.container(height=300):
                        st.dataframe(scoring_df, use_container_width=True)
                        
                    # Chuyển đổi để download
                    csv_export = scoring_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="⬇️ Tải xuống kết quả (CSV)",
                        data=csv_export,
                        file_name='ket_qua_du_bao_5C.csv',
                        mime='text/csv'
                    )
