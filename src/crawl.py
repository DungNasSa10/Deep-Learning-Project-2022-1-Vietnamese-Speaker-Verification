from crawling.pipeline import Pipeline
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath_or_url", type=str, required=True, help="Can be file path or google drive shared url")
    parser.add_argument("-d", "--save_dir", type=str, default="./data/wavs", help="Directory to save .wav files")
    parser.add_argument("-sr", "--sampling_rate", type=int, default=16000, help="Sampling rate of output .wav files")
    parser.add_argument("-rm", "--remove_mp3", type=bool, default=True, help="Remove downloaded mp3")
    args = parser.parse_args()

    p = Pipeline()
    p.run(**vars(args))