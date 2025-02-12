import streamlit as st
import Results
import Video
import mysql.connector
from streamlit_option_menu import option_menu


# Menghubungkan aplikasi ke database MySQL
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="streamlit"
)
# Membuat objek cursor untuk mengeksekusi query SQL
mycursor = mydb.cursor()

# Mengonfigurasi halaman pada Framework Streamlit
st.set_page_config(page_title="Helmet Detection", page_icon="👮‍♂️", layout="wide", initial_sidebar_state="expanded")


# Fungsi utama untuk menjalankan Framework Streamlit
def main():
    # Membuat sidebar
    with st.sidebar:
        app = option_menu(
            menu_title='Menu',
            options=['Video', 'Results'],
            icons=['camera-reels-fill', 'book-half'],
            menu_icon="list-ul",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": '#151515'},
                "icon": {"color": "white", "font-size": "23px"},
                "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px",
                             "--hover-color": "#3F474E"},
                "nav-link-selected": {"background-color": "#2569A5"}
            }
        )

    # Menjalankan bagian web berdasarkan menu yang dipilih
    if app == "Video":
        Video.app(mydb, mycursor)
    if app == "Results":
        Results.app(mycursor)


if __name__ == '__main__':
    main()
