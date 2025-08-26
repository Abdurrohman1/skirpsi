import streamlit as st
from PIL import Image
from src.predictor import predict_image

st.set_page_config(page_title='Brain Tumor Diagnosis', layout='centered')
#st.title('Brain Tumor Classification With VGG16 Model')#
st.markdown("<h1 style='text-align: center;'>Brain Tumor Classification With VGG16 Model</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1,1])

with col1 :
    st.markdown('#### Image Preview')
    uploaded_file =st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    else :
        st.warning('Silahkan Upload image terlebih dahulu')


with col2:
    st.markdown('####')
    predict_btn = st.button('Proses')
    if predict_btn:
        if uploaded_file:
            with st.spinner('Menganalisis Image . . . .'):
                prediction, confidence = predict_image(image)

            
            st.success('Prediction Result')
            
            st.markdown(f"<h2 style='text-align: center; color: green;'>{prediction}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>Tingkat Kepercayaan: <b>{confidence:.2%}</b></p>", unsafe_allow_html=True)
        else:
            st.error('Upload Gambar Terlebih Dahulu')

            