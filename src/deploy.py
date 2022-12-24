import streamlit as st
import subprocess
import os
import time
import hashlib
import random


def remove_file(file):
    if os.path.exists(file):
        os.remove(file)


def get_id(file_path):
    dr, fn = os.path.split(file_path)
    _hash = hashlib.sha256(f"{str(time.time())} {fn} {str(random.randint(1, 1000000))}".encode('utf-8')).hexdigest()
    p = os.path.join(dr, _hash[:16] + "-" + fn)
    return p


def create_test_list(wav_paths: list, output_file: str):
    assert len(wav_paths) > 1

    with open(output_file, 'w', encoding='utf-8') as f:
        size = len(wav_paths)
        for i in range(0, size - 1):
            for j in range(1, size):
                f.write(wav_paths[i] + " " + wav_paths[j] + "\n")
    f.close()


def predict(wav_paths: list):
    if len(wav_paths) <= 1:
        st.warning("You must choose more audio files and remove duplicated files if exist")
        return

    if "test_list" in st.session_state:
        remove_file(st.session_state.test_list)

    test_list = "data/tmp/test_list.txt"
    test_list = get_id(test_list)
    create_test_list(wav_paths, test_list)
    st.session_state.test_list = test_list

    # subprocess.check_call([
    #     'python src/trainer.py', 
    #     "--test",
    #     "--model", "ResNet50 ",
    #     "--encoder_type", "ASP ",
    #     "--initial_model", "output/ckpt/resnet50/model000000028.model",
    #     "--test_list", test_list,
    #     "--test_path", "data/tmp/wavs",
    #     "--output_path", "output/result/resnet50",
    # ],  shell=True)

    #### REmove all here


with st.form("my-form", clear_on_submit=True):
    uploaded_files = st.file_uploader("Choose audio", accept_multiple_files=True, type=['.wav', '.mp3'], key='files')

    if "wav_paths" not in st.session_state:
        st.session_state.wav_paths = []

    if len(uploaded_files) > 0:
        for wp in st.session_state.wav_paths:
            remove_file(wp)

        st.session_state.wav_paths = []

        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.getvalue()
            tmp_dir = "data/tmp/wavs"
            os.makedirs(tmp_dir, exist_ok=True)
            path = os.path.join(tmp_dir, uploaded_file.name)
            path = get_id(path)

            with open(path, 'wb') as f:
                f.write(bytes_data)
            f.close()

            st.session_state.wav_paths.append(path)

    submitted = st.form_submit_button("Predict!")

    if submitted:
        predict(st.session_state.wav_paths)

