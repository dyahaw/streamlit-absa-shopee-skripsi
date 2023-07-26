from nlp_id.lemmatizer import Lemmatizer
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import re
import numpy as np
import pandas as pd
import pickle

# preprocessing


def casefolding(content):
    content = content.casefold()
    return content


# Remove Puncutuation dan karakter yg tdk dibutuhkan
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-zA-Z]')


def clean_punct(content):
    content = clean_spcl.sub(' ', content)
    content = clean_symbol.sub(' ', content)
    content = re.sub(
        '((www\.[^\s]+) | (https?://[^\s]+))', ' ', content)  # remove url
    content = re.sub('@[^\s]+', ' ', content)  # remove username
    content = re.sub(':v', ' ', content)  # menghilangkan :v
    content = re.sub(';v', ' ', content)  # menghilangkan ;v
    # mengganti dgn sampai dengan
    content = re.sub('s/d', ' sampai dengan', content)
    return content


def replaceThreeOrMore(content):
    # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gool).
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", content)


# Kamus alay
alay_dict = pd.read_csv('new_kamusalay1.csv', names=[
                        'original', 'replacement'], encoding='latin-1')
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
# Mengganti kata-kata yang dianggap alay


def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])


def removeangkachar(content):
    content = re.sub('\d+', '', content)  # Remove angka
    return content.strip(" ")
# Menghapus Double atau Lebih Whitespace


def normalize_whitespace(content):
    content = str(content)
    content = re.sub(r"//t", r"\t", content)
    content = re.sub(r"( )\1+", r"\1", content)
    content = re.sub(r"(\n)\1+", r"\1", content)
    content = re.sub(r"(\r)\1+", r"\1", content)
    content = re.sub(r"(\t)\1+", r"\1", content)
    return content.strip(" ")


nltk.download('punkt')


def tokenisasi(content):
    content = nltk.tokenize.word_tokenize(content)
    return content


factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
exclude_stopword = ['tidak', 'belum', 'selain',
                    'tidak bisa', 'bisa', 'belum bisa', 'ada']
remove_words = ([word for word in stopwords if word not in exclude_stopword])
final_stop_words = ArrayDictionary(remove_words)


def remove_stopwords(content):
    factory = StopWordRemover(final_stop_words)
    content = [factory.remove(x) for x in content]
    return content


def lemma_preproc(content):
    lemmatizer = Lemmatizer()
    content = [lemmatizer.lemmatize(x) for x in content]
    return content


def token_kosong(content):
    content = [x for x in content if len(x) > 0]
    return content


def jointeks(content):
    for i in content:
        return " ".join(content)


def preprocess(content):
    content = casefolding(content)
    content = clean_punct(content)
    content = replaceThreeOrMore(content)
    content = normalize_alay(content)
    content = removeangkachar(content)
    content = normalize_whitespace(content)
    content = tokenisasi(content)
    content = remove_stopwords(content)
    content = lemma_preproc(content)
    content = token_kosong(content)
    content = jointeks(content)
    return content


# load save model
model_aspek = pickle.load(open('model_aspek.sav', 'rb'))
model_sentimen = pickle.load(open('model_sentimen.sav', 'rb'))
model_aspek_file = pickle.load(open('model_aspek.sav', 'rb'))
model_sentimen_file = pickle.load(open('model_sentimen.sav', 'rb'))
tf_idf_data = pickle.load(open('tf_idf_data.sav', 'rb'))
data = pickle.load(open('data.sav', 'rb'))
data_file = data
tf_idf_data_file = tf_idf_data


# judul halaman
# st.title("Aspect Based Sentimen Analysis Pada Ulasan")


st.markdown("<h2 style='text-align: center; color: system;'>Aspect Based Sentiment Analysis Pada Ulasan</h2>",
            unsafe_allow_html=True)
st.markdown("<hr></hr>", unsafe_allow_html=True)

####### Glosarium #######
st.markdown("#### Glosarium")
with st.expander("📗 -  Glosarium", expanded=False):
    st.write("""
    - ##### _E-Service Quality_ (Kualitas Layanan Elektronik)
        - Menurut (Chong & Man, 2017), E-service quality atau kualitas layanan secara elektronik adalah proses menciptakan nilai lebih pada suatu produk sehingga suatu produk mendapatkan nilai tambah dari konsumen dan mempertahankan citra suatu perusahaan.
        - E-Service Quality sangat dibutuhkan dalam e-commerce agar bisnis dapat berjalan dengan baik dan mendapatkan kepercayaan dari pelanggan.
        - Penilaian konsumen terhadap suatu e-commerce tidak hanya dapat dilihat melalui interaksi dengan suatu situs, melainkan dengan aspek pelayanannya.
        - E-Service Quality dapat dilakukan dengan memberikan kenyamanan, kecepatan, dan kemudahan pada konsumen.
        - Terdapat 7 dimensi pendekatan yang digunakan dalam mengukur kualitas pelayanan dengan metode E-S-QUAL dan E-RecS-QUAL yang dikembangkan oleh Parasurman, diantaranya adalah efficiency, fullfilment, system availability, privacy, responsiveness, compensation, dan contact (Çelik, 2021).
            - (Çelik, 2021)\t:  https://doi.org/10.15295/bmij.v9i3.1898
        - Pada penelitian ini, label aspek yang digunakan berdasarkan 7 dimensi pengukuran kualitas pelayanan dengan metode E-S-QUAL dan E-RecS-QUAL yang dikembangkan Parasurman.
    """)
    st.write("\n\n")
    st.write("""
    - ##### Label Kategori Kelas Aspek
        - Aspek yang digunakan pada penelitian ini, terdiri dari 7 aspek antara lain yaitu:
            - _Efficiency_
                - Berkaitan dengan kecepatan dan kemudahan dalam mengakses dan menggunakan suatu situs.
                - Contohnya: lama dalam memuat halaman, mudah dalam mengakses atau masuk ke dalam aplikasi, memiliki banyak metode pembayaran sehingga memudahkan dalam melakukan transaksi.
            - _Fulfillment_
                - Berkaitan dengan ketersediaan dan pengiriman produk.
                - Contohnya: produk atau layanan yang dikirimkan tiba dengan aman atau rusak, produk tiba sesuai dengan estimasi waktu yang telah ditentukan, barang memiliki stok/tersedia sesuai dengan aplikasi.
            - _System Availability_
                - Berkaitan dengan fungsi teknis pada suatu situs yang dapat berjalan lancar dan benar.
                - Contohnya: fitur login dapat berfungsi dengan baik dan tidak eror, ketika melakukan klik ikon _love_ untuk menyimpan yang disukai dapat tersimpan dengan baik.
            - _Privacy_
                - Berkaitan dengan keamanan suatu aplikasi untuk dapat melindungi dan mengamankan informasi data pelanggan suatu situs dari penipuan dan kebocoran informasi pribadi pelanggan yang menggunakan aplikasi tersebut.
                - Contohnya: Data dapat terlindungi dengan baik dan tidak ada kebocoran data.
            - _Responsiveness_
                - Berkaitan dengan kemampuan dalam memberikan tanggapan/respons atau menangani dengan cepat saat pelanggan mengalami masalah.
                - Contohnya: respons customer service yang lambat ketika dichat, tanggapan dari permasalahan berbelit belit sehingga tidak menemukan solusi, informasi terkait laporan keluhan yang ditanggapi tidak konsisten sehingga permasalahan tidak cepat selesai.
            - _Compensation_
                - Berkaitan dengan kemampuan dalam memberikan kompensasi berupa pengembalian biaya pengiriman dan biaya layanan ketika terjadi masalah dan memberikan layana _return_ barang saat barang yang dikirim rusak.
                - Contohnya: Dapat melakukan pengembalian barang atau pengembalian biaya saat produk rusak.
            - _Contact_
                - Berkaitan dengan kemampuan dalam menghubungi pusat layanan atau kontak bantuan atau kontak kurir saat dibutuhkan/mengalami masalah dan informasi kontak terdapat pada suatu situs aplikasi.
                - Contohnya: Dapat menghubungi pusat bantuan karena informasi kontak yang mudah didapatkan, suatu aplikasi menyediakan _call center_ 24 jam.
        - Sumber:
            - (Çelik, 2021)\t:  https://doi.org/10.15295/bmij.v9i3.1898
    """)
    st.write("\n\n")
    st.write("""
    - ##### Label Kategori Kelas Sentimen
        - Polaritas sentimen mengacu pada arah emosi atau penilaian yang terkandung dalam teks.
        - Kelas sentimen yang digunakan pada penelitian ini antara lain yaitu:
            - Positif
            Ulasan yang mengandung  mengandung kata-kata, emosi, dan dukungan yang positif baik secara implisit maupun eksplisit.
            - Negatif
            Ulasan yang mengandung kata-kata negatif, emosi, dan dukungan negatif baik secara implisit maupun eksplisit.
    """)
    st.write("\n\n")
st.markdown("<hr></hr>", unsafe_allow_html=True)

####### Prediksi #######
st.markdown("#### Prediksi Kelas Aspek dan Sentimen pada Ulasan (Kalimat)")
text_input = st.text_input("Masukkan Kalimat Ulasan")
review = {'review': [text_input]}
new_data = pd.DataFrame(review)
new_data['preprocess'] = new_data['review'].apply(preprocess)
new_data = new_data.loc[:, ['preprocess']]
new_data = new_data.rename(columns={"preprocess": "review"})


def Tokenize(data):
    data['review_token'] = ""
    data['review'] = data['review'].astype('str')
    for i in range(len(data)):
        data['review_token'][i] = data['review'][i].lower().split()
    all_tokenize = sorted(
        list(set([item for sublist in data['review_token'] for item in sublist])))
    return data, all_tokenize


def tf(data, all_tokenize):
    from operator import truediv
    token_cal = Tokenize(data)
    data_tokenize = token_cal[0]
    for item in all_tokenize:
        data_tokenize[item] = 0
    for item in all_tokenize:
        for i in range(len(data_tokenize)):
            if data_tokenize['review_token'][i].count(item) > 0:
                a = data_tokenize['review_token'][i].count(item)
                b = len(data_tokenize['review_token'][i])
                c = a / b
                data_tokenize[item] = data_tokenize[item].astype('float')
                data_tokenize[item][i] = c
    return data_tokenize


def tfidf_shopee(data, new_data=new_data, tf_idf_data=tf_idf_data):
    tf_idf = tf_idf_data
    N = len(data)
    all_tokenize = tf_idf.columns.tolist()
    df = {}
    for item in all_tokenize:
        df_ = (tf_idf[item] > 0).sum()
        df[item] = df_
        idf = (np.log(N / df_)) + 1
        tf_idf[item] = tf_idf[item] * idf

    if new_data is not None:
        new_tf = tf(new_data, all_tokenize)

        for item in all_tokenize:
            if item in new_tf.columns:
                df_ = df.get(item, 0)
                idf = (np.log(N / (df_ + 1))) + 1
                new_tf[item] = new_tf[item] * idf

        new_tf.drop(columns=['review', 'review_token'], inplace=True)

        return new_tf, df
    else:
        return tf_idf, df


tfidf_result, document_frequency = tfidf_shopee(data, new_data)

# aspect


def testing_aspek(W_aspek, data_uji_aspek):
    prediksi_aspek = np.array([])
    for i in range(data_uji_aspek.shape[0]):
        y_prediksi_aspek = np.sign(
            np.dot(W_aspek, data_uji_aspek.to_numpy()[i]))
        prediksi_aspek = np.append(prediksi_aspek, y_prediksi_aspek)
    return prediksi_aspek


def testing_onevsrest_aspek(W_aspek, data_uji_aspek):
    list_kelas_aspek = W_aspek.keys()
    hasil_aspek = pd.DataFrame(columns=W_aspek.keys())
    for kelas_aspek in list_kelas_aspek:
        hasil_aspek[kelas_aspek] = testing_aspek(
            W_aspek[kelas_aspek], data_uji_aspek)
    kelas_prediksi_aspek = hasil_aspek.idxmax(1)
    return kelas_prediksi_aspek


prediksi_aspek = testing_onevsrest_aspek(model_aspek, new_data)

# sentimen


def testing_sentimen(W_sentimen, data_uji_sentimen):
    prediksi_sentimen = np.array([])
    for i in range(data_uji_sentimen.shape[0]):
        y_prediksi_sentimen = np.sign(
            np.dot(W_sentimen, data_uji_sentimen.to_numpy()[i]))
        prediksi_sentimen = np.append(
            prediksi_sentimen, y_prediksi_sentimen)
    return prediksi_sentimen


y_prediksi_sentimen = testing_sentimen(model_sentimen, new_data)

prediksi = st.button("Hasil Prediksi")

if prediksi:
    for aspek, sentimen in zip(prediksi_aspek, y_prediksi_sentimen):
        st.success(
            f"Aspek {aspek}, Sentimen {'Positif' if sentimen == 1 else 'Negatif'}")

st.markdown("<hr></hr>", unsafe_allow_html=True)


####### Prediksi #######
st.markdown("#### Prediksi Kelas Aspek dan Sentimen Ulasan File (CSV)")
data_file_csv = st.file_uploader("Upload File CSV", type=['csv'])
if data_file_csv is not None:
    file_detail = {"Filename": data_file_csv.name,
                   "FileType": data_file_csv.type, "FileSize": data_file_csv.size}
    st.write(file_detail)
    df = pd.read_csv(data_file_csv, encoding='unicode_escape',
                     low_memory=False)
    st.dataframe(df)

    new_data_file = pd.DataFrame(df)
    new_data_file['preprocess'] = new_data_file['review'].apply(preprocess)
    new_data_file = new_data_file.loc[:, ['preprocess']]
    new_data_file = new_data_file.rename(columns={"preprocess": "review"})

    def Tokenize_file(data_file):
        data_file['review_token'] = ""
        data_file['review'] = data_file['review'].astype('str')
        for i_file in range(len(data_file)):
            data_file['review_token'][i_file] = data_file['review'][i_file].lower(
            ).split()
        all_tokenize_file = sorted(list(set(
            [item_file for sublist in data_file['review_token'] for item_file in sublist])))
        return data_file, all_tokenize_file

    def tf_file(data_file, all_tokenize_file):
        from operator import truediv
        token_cal_file = Tokenize_file(data_file)
        data_tokenize_file = token_cal_file[0]
        for item_file in all_tokenize_file:
            data_tokenize_file[item_file] = 0
        for item_file in all_tokenize_file:
            for i_file in range(len(data_tokenize_file)):
                if data_tokenize_file['review_token'][i_file].count(item_file) > 0:
                    a_file = data_tokenize_file['review_token'][i_file].count(
                        item_file)
                    b_file = len(data_tokenize_file['review_token'][i_file])
                    c_file = a_file / b_file
                    data_tokenize_file[item_file] = data_tokenize_file[item_file].astype(
                        'float')
                    data_tokenize_file[item_file][i_file] = c_file
        return data_tokenize_file

    def tfidf_shopee_file(data_file, new_data_file=new_data_file, tf_idf_data_file=tf_idf_data_file):
        tf_idf_file = tf_idf_data_file
        N_file = len(data_file)
        all_tokenize_file = tf_idf_file.columns.tolist()
        df_file = {}
        for item_file in all_tokenize_file:
            df_file_ = (tf_idf_file[item_file] > 0).sum()
            df_file[item_file] = df_file_
            idf_file = (np.log(N_file / df_file_)) + 1
            tf_idf_file[item_file] = tf_idf_file[item_file] * idf_file

        if new_data_file is not None:
            new_tf_file = tf_file(new_data_file, all_tokenize_file)

            for item_file in all_tokenize_file:
                if item_file in new_tf_file.columns:
                    df_file_ = df_file.get(item_file, 0)
                    idf_file = (np.log(N_file / (df_file_ + 1))) + 1
                    new_tf_file[item_file] = new_tf_file[item_file] * idf_file

            new_tf_file.drop(columns=['review', 'review_token'], inplace=True)

            return new_tf_file, df_file
        else:
            return tf_idf_file, df_file

    tfidf_shopee_file(data_file, new_data_file)

# aspect

    def testing_aspek_file(W_aspek_file, data_uji_aspek_file):
        prediksi_aspek_file = np.array([])
        for i_file in range(data_uji_aspek_file.shape[0]):
            y_prediksi_aspek_file = np.sign(
                np.dot(W_aspek_file, data_uji_aspek_file.to_numpy()[i_file]))
            prediksi_aspek_file = np.append(
                prediksi_aspek_file, y_prediksi_aspek_file)
        return prediksi_aspek_file

    def testing_onevsrest_aspek_file(W_aspek_file, data_uji_aspek_file):
        list_kelas_aspek_file = W_aspek_file.keys()
        hasil_aspek_file = pd.DataFrame(columns=W_aspek_file.keys())
        for kelas_aspek_file in list_kelas_aspek_file:
            hasil_aspek_file[kelas_aspek_file] = testing_aspek_file(
                W_aspek_file[kelas_aspek_file], data_uji_aspek_file)
        kelas_prediksi_aspek_file = hasil_aspek_file.idxmax(1)
        return kelas_prediksi_aspek_file

    prediksi_aspek_file = testing_onevsrest_aspek_file(
        model_aspek_file, new_data_file)

# sentimen

    def testing_sentimen_file(W_sentimen_file, data_uji_sentimen_file):
        prediksi_sentimen_file = np.array([])
        for i_file in range(data_uji_sentimen_file.shape[0]):
            y_prediksi_sentimen_file = np.sign(
                np.dot(W_sentimen_file, data_uji_sentimen_file.to_numpy()[i_file]))
            prediksi_sentimen_file = np.append(
                prediksi_sentimen_file, y_prediksi_sentimen_file)
        return prediksi_sentimen_file

    y_prediksi_sentimen_file = testing_sentimen_file(
        model_sentimen_file, new_data_file)


prediksi_file = st.button("Hasil Prediksi File")

if prediksi_file:
    data = []
    for aspek, sentimen in zip(prediksi_aspek_file, y_prediksi_sentimen_file):
        data.append({
            'Aspek': aspek,
            'Sentimen': 'Positif' if sentimen == 1 else 'Negatif'
        })
    # Create a DataFrame from the list of dictionaries
    st.success('Hasil Prediksi:')
    df = pd.DataFrame(data)
    df
    st.download_button(label='Download File CSV',
                       data=df.to_csv(), mime='text/csv')

st.markdown("<hr></hr>", unsafe_allow_html=True)
