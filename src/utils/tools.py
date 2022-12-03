import pandas as pd
import os


__gg_drive_downloadable_prefix  = "https://drive.google.com/uc?id="
__gg_drive_prefix               = "https://drive.google.com"
__gg_drive_shared_link_postfix  = "usp=share_link"


def get_downloadable_gg_drive_url(raw_url: str):
    return __gg_drive_downloadable_prefix + raw_url.split("/")[-2]


def is_gg_drive_url(url: str):
    return url.startswith(__gg_drive_prefix)


def is_gg_drive_url_downloadable(url: str):
    return url.startswith(__gg_drive_downloadable_prefix)


def get_voices_and_urls(filepath_or_url: str) -> list:
    """
    Args:
        filepath: str
            filepath
    
    Returns:
        voice names, together with url lists
    """
    if is_gg_drive_url(filepath_or_url):
        if is_gg_drive_url_downloadable(filepath_or_url) is False:
            filepath_or_url = get_downloadable_gg_drive_url(filepath_or_url)
    elif os.path.exists(filepath_or_url) is False:
        raise ValueError(f"Cannot detect file or url: {filepath_or_url}")

    df = pd.read_csv(filepath_or_url, header=1).dropna()
    df = df[df['Crawled'] == False]
    df['urls'] = df['Playlist URL'].apply(lambda x: x.split())
    df['Voice'] = df['Voice'].apply(lambda x: x.strip())
    return df[['Voice', 'urls']].values.tolist()


def get_wav_files(wav_dir: str) -> list:
    for root, _, files in os.walk(wav_dir):
        return [os.path.join(root, file) for file in files if os.path.splitext(file)[-1] == '.wav']
