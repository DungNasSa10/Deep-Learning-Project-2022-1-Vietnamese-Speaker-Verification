import streamlit as st
import os
import time
import hashlib
import random

from crawling.downloader import Downloader

from learning.tasks.infer import infer


def remove_file(file):
    if file is None:
        return

    if os.path.exists(file):
        os.remove(file)


def get_id(file_path):
    dr, fn = os.path.split(file_path)
    _hash = hashlib.sha256(f"{str(time.time())} {fn} {str(random.randint(1, 1000000))}".encode('utf-8')).hexdigest()
    p = os.path.join(dr, _hash[:16] + "-" + fn)
    return p.replace("\\", "/")


def predict(model_names, first_path: list, second_path):
    if first_path is None or second_path is None:
        st.warning("You must choose two audio files")
        remove_file(first_path)
        remove_file(second_path)
        return

    for name in model_names:
        prob = infer(model=name, 
            pretrained_checkpoint=f"output/ckpt/{name}.model",
            wav_path_1=first_path,
            wav_path_2=second_path)

        st.write(f"Probability for {name}: {prob}")

    remove_file(first_path)
    remove_file(second_path)


def upfile_on_change(keys):
    for key in keys:
        if key in st.session_state:
            remove_file(st.session_state[key])
            del st.session_state[key]


tmp_dir = "data/tmp/wavs"
os.makedirs(tmp_dir, exist_ok=True)
keys = ['upfile0', 'upfile1']


def upload_audio(msg, key, keys):
    upfile = st.file_uploader(msg, type=['.wav', '.mp3'], on_change=upfile_on_change, args=(keys,))
    if upfile is not None:
        bytes_data = upfile.getvalue()
        path = os.path.join(tmp_dir, upfile.name)
        path.replace("\\", "/")
        path = get_id(path)

        with open(path, 'wb') as f:
            f.write(bytes_data)
        f.close()

        if os.path.splitext(path) == ".mp3":
            wav_path = Downloader.mp3_to_wav(path)
            os.remove(path)
            path = wav_path

        st.session_state[key] = path

        st.audio(bytes_data)
  

upload_audio("Choose first audio", key=keys[0], keys=keys)
upload_audio("Choose second audio", key=keys[1], keys=keys)

st.write("Choose your model")
model_names = [
    'SEResNet34',
    'Rawnet3',
]

options = [st.checkbox(name) for name in model_names]

submitted = st.button("Predict!")
if submitted:
    predict([name for name, op in zip(model_names, options) if op], *[st.session_state.get(k, None) for k in keys])

