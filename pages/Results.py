import streamlit as st
from functions import function_system
from moviepy.editor import VideoFileClip


# Fungsi utama untuk menampilkan halaman Results
def app(mycursor):
    # Menampilkan judul halaman
    st.title("Results Page")

    # Query untuk mengambil nilai 'datetime' dari tabel 'results' di database
    sql_datetime = "Select datetime from results"
    mycursor.execute(sql_datetime)
    data = mycursor.fetchall()
    unique_data = function_system.fix_array(data)

    # Path ke folder results
    results_path = '../results'

    # Membuat container di Streamlit untuk menampilkan hasil video
    container_result_video = st.container(border=1)

    # Menampilkan konten dalam container
    with container_result_video:
        # Membuat dropdown untuk memilih folder berdasarkan data unik yang diambil dari database
        option_folder = st.selectbox("Select a Folder", unique_data, index=None, placeholder="Select a Folder File...")

        if option_folder:
            # Menentukan direktori gambar dan video berdasarkan folder yang dipilih
            image_dir = f'{results_path}/{option_folder}/images'
            video_dir = f'{results_path}/{option_folder}/videos/result_video.mp4'

            # Menggunakan moviepy untuk membuka dan memproses video
            video = VideoFileClip(video_dir)
            video.write_videofile(f'{results_path}/{option_folder}/videos/output_video.mp4', codec="libx264", audio_codec="aac")
            video_dir_output = f'{results_path}/{option_folder}/videos/output_video.mp4'

            # Query untuk mengambil nama file gambar berdasarkan 'datetime' yang sesuai dengan folder yang dipilih
            sql_file = f'Select image from results where datetime = "{option_folder}"'
            mycursor.execute(sql_file)
            files = mycursor.fetchall()
            fix_files = function_system.fix_array(files)

            # Membuat array untuk menyimpan path lengkap gambar
            images_path_array = []
            for file in fix_files:
                images_path_array.append(image_dir + '/' + file)

            # Menampilkan subjudul dan video hasil output
            st.subheader("Results")
            st.video(video_dir_output)

            # Menentukan jumlah kolom untuk menampilkan gambar
            num_columns = 5
            cols_image = st.columns(num_columns)

            # Jika tidak ada gambar di folder, menampilkan pesan bahwa tidak ada gambar
            if len(images_path_array) == 0:
                no_img_html = """
                <div style="text-align: center"> No Image Result In This Folder </div>
                """

                st.markdown(no_img_html, unsafe_allow_html=True)
            # Jika ada gambar, menampilkan gambar dalam beberapa kolom
            else:
                for i, image_path in enumerate(images_path_array):
                    with cols_image[i % num_columns]:
                        st.image(image_path, caption=f'Result Image {i + 1}', use_column_width=True)
