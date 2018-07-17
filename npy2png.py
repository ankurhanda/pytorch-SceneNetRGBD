import argparse
import cv2
import numpy as np
import pathlib
import sys
import os

def convert_files(infolder,outfolder):
    if not os.path.exists(infolder) or not os.path.isdir(infolder):
        print('Input folder {0} does not exist'.format(infolder))
        sys.exit(1)
    if not os.path.exists(outfolder) or not os.path.isdir(outfolder):
        if outfolder == 'converted_nyu_data':
            os.mkdir('converted_nyu_data')
        else:
            print('Output folder {0} does not exist'.format(outfolder))
            sys.exit(1)
    for p in pathlib.Path(infolder).iterdir():
        if str(p).endswith('rgb.npy'):
            numpy_rgb = np.load(str(p))
            cv2_rgb = np.uint8(np.transpose(numpy_rgb,(1,2,0))*255.0)
            cv2_bgr = cv2.cvtColor(cv2_rgb, cv2.COLOR_BGR2RGB)
            output_filename = os.path.join(outfolder,str(p.stem).split('_')[0]+'_rgb.png')
            cv2.imwrite(output_filename,cv2_bgr)
        elif str(p).endswith('depth.npy'):
            numpy_depth = np.load(str(p))[0]*1000.0
            output_filename = os.path.join(outfolder,str(p.stem).split('_')[0]+'_depth.png')
            cv2.imwrite(output_filename,np.uint16(numpy_depth))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Numpy to png converter')
    parser.add_argument('--infolder',help='The folder containing pre-scaled rgb and depth numpy files',
                        default='nyu_data'
                        )
    parser.add_argument('--outfolder',help='The folder to output converted rgb and depth files',
                        default='converted_nyu_data'
                        )
    args = parser.parse_args()
    convert_files(args.infolder,args.outfolder)
