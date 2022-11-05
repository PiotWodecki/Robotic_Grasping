import argparse
import glob
import os
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove unnecessary *_stereo_depth.tiff')
    parser.add_argument('path',  nargs= '+', help='Path to Jacquard Grasping Dataset')
    args, unknown = parser.parse_known_args()
    path = ''.join(args.path)
    path = path.replace("'", '')

    files = glob.glob(os.path.join(path, '*', '*stereo_depth.tiff'))
    for filename in files:
        os.remove(filename)