import librosa
import numpy
import os
import logging
import csv
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(message)s',
                    handlers=(logging.FileHandler('working.log', 'w'), logging.StreamHandler(sys.stdout),))

TEST_SOUNDTRACK_DIRECTORY = './test_soundtracks/'
RESULT_CSV_PATH = 'result.csv'


def get_tracks_files(dir: str) -> list:
    return [os.path.join(dir, file) for file in os.listdir(dir)]


def get_track_data(filename: str) -> dict:
    ys, sr = librosa.load(filename, offset=3, duration=10)

    spec_centroid = numpy.average(librosa.feature.spectral_centroid(ys, sr))
    spec_bandwidth = numpy.average(librosa.feature.spectral_bandwidth(ys, sr))
    spec_rolloff = numpy.average(librosa.feature.spectral_rolloff(ys, sr))
    zero_crossing_rate = numpy.average(librosa.feature.zero_crossing_rate(ys))
    mfcc = numpy.average(librosa.feature.mfcc(ys, sr, dct_type=1))

    return {"spectral centroid avg": spec_centroid, "spectral bandwidth avg": spec_bandwidth,
            "spectral rolloff avg": spec_rolloff, "zero crossing rate avg": zero_crossing_rate,
            "mfcc": mfcc}


def main():
    logging.info('starting program')
    files = get_tracks_files(TEST_SOUNDTRACK_DIRECTORY)
    logging.info('got %d files from %s dir', len(files), TEST_SOUNDTRACK_DIRECTORY)

    result = []
    for file in files:
        current_dict = {"filename": file}
        current_dict.update(get_track_data(file))
        result.append(current_dict)
        logging.debug('added new data to total results. current progress %.2f', (len(result) / len(files) * 100))

    logging.info('total length %d', len(result))
    save_data_to_csv(result, RESULT_CSV_PATH)
    logging.info('successfully saved results to csv %s', RESULT_CSV_PATH)


def save_data_to_csv(data, path):
    with open(path, 'w') as f:
        writer = csv.DictWriter(f, list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


if __name__ == '__main__':
    main()
