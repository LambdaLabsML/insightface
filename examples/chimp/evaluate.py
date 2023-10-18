import os
import argparse
import pickle
import numpy as np


def compute_std(data):
    
    data_sum = np.sum(data, axis=(1, 2))
    idx_nonzero = np.nonzero(data_sum)
    
    if len(idx_nonzero) > 0:
        data = data[idx_nonzero] 
        av = np.mean(np.std(data, axis=0))
    else:
        av = -1.0
    return av

def main(args):
    file_list_transform = os.path.join(args.input_dir, 'list_transform' + '.pickle')
    file_list_image = os.path.join(args.input_dir, 'list_image' + '.pickle')

    with open(file_list_transform, 'rb') as handle:
        list_transform = pickle.load(handle)

    with open(file_list_image, 'rb') as handle:
        list_image = pickle.load(handle)

    av_image = {}
    for img in list_image:
        av_trans = {}
        for trans in list_image[img]:
            av_trans[trans] = compute_std(list_image[img][trans])
        av_image[img] = av_trans
    
    print(av_image)

    with open(os.path.join(args.input_dir, 'std.pickle'), 'wb') as handle:
        pickle.dump(av_image, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", help="Input folder", dest="input_dir", type=str)

    args = parser.parse_args()
    main(args)