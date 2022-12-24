import streamlit as st
import subprocess
import os


def upload_wavs():
    uploaded_files = st.file_uploader("Choose audio", accept_multiple_files=True, type=['.wav', '.mp3'])
    wav_paths = []

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.getvalue()
        tmp_dir = "data/tmp/wavs"
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, uploaded_file.name)

        with open(path, 'wb') as f:
            f.write(bytes_data)
        f.close()

        wav_paths.append(path)

    return wav_paths

        
wav_paths = upload_wavs()


def clean_duplicate(wav_paths):
    wav_paths = sorted(wav_paths)
    wav_paths.append(None)
    return [wav_paths[i] for i in range(len(wav_paths) - 1) if wav_paths[i] != wav_paths[i + 1]]


def create_test_list(wav_paths: list, output_file: str):
    assert len(wav_paths) > 1

    with open(output_file, 'w', encoding='utf-8') as f:
        size = len(wav_paths)
        for i in range(0, size - 1):
            for j in range(1, size):
                f.write(wav_paths[i] + " " + wav_paths[j] + "\n")
    f.close()


def predict(wav_paths: list):
    wav_paths = clean_duplicate(wav_paths)

    if len(wav_paths) <= 1:
        st.warning("You must choose more audio files and remove duplicated files if exist")
        return

    test_list = "data/tmp/test_list.txt"

    create_test_list(wav_paths, test_list)

    subprocess.check_call([
        'python src/trainer.py', 
        "--test",
        "--model", "ResNet50 ",
        "--encoder_type", "ASP ",
        "--initial_model", "output/ckpt/resnet50/model000000028.model",
        "--test_list", test_list,
        "--test_path", "data/tmp/wavs",
        "--output_path", "output/result/resnet50",
    ],  shell=True)


if st.button('Predict'):
    predict(wav_paths)

