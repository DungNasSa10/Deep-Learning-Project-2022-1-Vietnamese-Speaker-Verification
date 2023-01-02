import os
import time
import hashlib
import random
import streamlit as st

from crawling.downloader import Downloader
from learning.tasks.infer import infer


class Config:
    CHECKPOINT_DIR = "output/checkpoints"
    TEMPORARY_DIR = "data/tmp/wavs"
    AUDIO_KEYS = ['upfile0', 'upfile1']
    TABLE_WIDTH = 600


os.makedirs(Config.TEMPORARY_DIR, exist_ok=True)


def remove_file(file):
    if file is not None:
        if os.path.exists(file):
            os.remove(file)


def join_path(*args):
    path = os.path.join(*args)
    path.replace("\\", "/")
    return path


def get_id(file_path):
    dr, fn = os.path.split(file_path)
    _hash = hashlib.sha256(f"{str(time.time())} {fn} {str(random.randint(1, 1000000))}".encode('utf-8')).hexdigest()
    p = os.path.join(dr, _hash[:16] + "-" + fn)
    return p.replace("\\", "/")


def upload_audio(msg, key):
    upfile = st.file_uploader(msg, type=['.wav', '.mp3'], key=key)
    if upfile is not None:
        bytes_data = upfile.getvalue()
        st.audio(bytes_data)
  

def predict(model_names, audio_keys, table_canvas):
    for key in audio_keys:
        if st.session_state.get(key) is None:
            st.warning("You must choose two audio files")
            print("You must choose two audio files")
            return False

    results = {
        "Model": [],
        "Probability": []
    }
    audio_paths = []

    for key in audio_keys:
        upfile = st.session_state[key]
        bytes_data = upfile.getvalue()
        path = join_path(Config.TEMPORARY_DIR, upfile.name)
        path = get_id(path)

        with open(path, 'wb') as f:
            f.write(bytes_data)
        f.close()

        if os.path.splitext(path)[-1] == ".mp3":
            wav_path = Downloader.mp3_to_wav(path)
            os.remove(path)
            path = wav_path
        
        audio_paths.append(path)

    table_canvas.empty()
    pbar_canvas = st.empty()
    pbar = pbar_canvas.progress(0.0)

    try:
        for idx, name in enumerate(model_names):
            print("Model:", name)
            prob = infer(model=name, 
                pretrained_checkpoint=os.path.join(Config.CHECKPOINT_DIR, f"{name}.model"),
                wav_path_1=audio_paths[0],
                wav_path_2=audio_paths[1])

            results['Model'].append(name)
            results["Probability"].append(prob)
            
            pbar.progress((idx + 1) / len(model_names))
            table_canvas.dataframe(results, width=Config.TABLE_WIDTH)

        pbar_canvas.empty()
        st.session_state.results = results
    finally:
        for path in audio_paths:
            remove_file(path)

    return True


def main():
    st.write("## Demo for Deep Learning project of group 15")
    st.write("### Choose your audio files")
    upload_audio("Choose first audio",  Config.AUDIO_KEYS[0])
    upload_audio("Choose second audio", Config.AUDIO_KEYS[1])

    st.write("### Choose your models")
    model_names = [
        "ECAPA_CNN_TDNN",
        "ECAPA_TDNN",
        "RawNet3",
        "SEResNet34",
        "VGG_M_40",
    ]

    options = [st.checkbox(name) for name in model_names]

    submitted_btn = st.empty()
    table_canvas = st.empty()

    if "results" in st.session_state:
        table_canvas.dataframe(st.session_state.results, width=Config.TABLE_WIDTH)

    submitted = submitted_btn.button("Submit", key='1')
    if submitted:
        submitted_btn.button("Waiting", disabled=True, key='2')
        if predict(
            [name for name, op in zip(model_names, options) if op], 
            Config.AUDIO_KEYS, 
            table_canvas
        ):
            st.experimental_rerun()


main()