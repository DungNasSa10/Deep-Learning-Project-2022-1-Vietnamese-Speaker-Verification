import pandas as pd
import os


def get_voices_and_urls(filepath: str) -> list:
    """
    Args:
        filepath: str
            filepath
    
    Returns:
        voice names, together with url lists
    """
    df = pd.read_csv(filepath, header=1).dropna()
    df = df[df['Crawled'] == False]
    df['urls'] = df['Playlist URL'].apply(lambda x: x.split())
    df['Voice Name'] = df['Voice Name'].apply(lambda x: x.strip())
    return df[['Voice Name', 'urls']].values.tolist()


def get_wav_files(wav_dir: str) -> list:
    # return glob(os.path.join(wav_dir, '*.wav'))
    for root, _, files in os.walk(wav_dir):
        return [os.path.join(root, file) for file in files if os.path.splitext(file)[-1] == '.wav']