# v3.classification
# 3/12/2018

import argparse
import os
import glob
import numpy as np
import cv2
import torch
import sklearn.feature_extraction.image

from torchvision.models import DenseNet

#-----helper function to split data into batches
def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 

# ----- parse command line arguments
parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",
                    nargs="*")

parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 10", default=10, type=int)
parser.add_argument('-o', '--outdir', help="outputdir, default ./output/", default="./output/", type=str)
parser.add_argument('-r', '--resize', help="resize factor 1=1x, 2=2x, .5 = .5x", default=1, type=float)
parser.add_argument('-m', '--model', help="model", default="best_model.pth", type=str)
parser.add_argument('-i', '--gpuid', help="id of gpu to use", default=0, type=int)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
parser.add_argument('-b', '--basepath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="", type=str)

args = parser.parse_args()

if not (args.input_pattern):
    parser.error('No images selected with input pattern')

OUTPUT_DIR = args.outdir
resize = args.resize

batch_size = args.batchsize
patch_size = args.patchsize
stride_size = patch_size//2


# ----- load network
device = torch.device(args.gpuid if args.gpuid!=-2 and torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                 num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                 drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["num_classes"]).to(device)

model.load_state_dict(checkpoint["model_dict"])
model.eval()

print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

# ----- get file list

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = []
basepath = args.basepath  #
basepath = basepath + os.sep if len(
    basepath) > 0 else ""  # if the user supplied a different basepath, make sure it ends with an os.sep

if len(args.input_pattern) > 1:  # bash has sent us a list of files
    files = args.input_pattern
elif args.input_pattern[0].endswith("tsv"):  # user sent us an input file
    # load first column here and store into files
    with open(args.input_pattern[0], 'r') as f:
        for line in f:
            if line[0] == "#":
                continue
            files.append(basepath + line.strip().split("\t")[0])
else:  # user sent us a wildcard, need to use glob to find files
    files = glob.glob(args.basepath + args.input_pattern[0])


# ------ work on files
for fname in files:

    fname = fname.strip()
    newfname_class = "%s/%s_class.png" % (OUTPUT_DIR, os.path.basename(fname)[0:-4])

    print(f"working on file: \t {fname}")

    # if not args.force and os.path.exists(newfname_class):
    #     print("Skipping as output file exists")
    #     continue
    #
    # cv2.imwrite(newfname_class, np.zeros(shape=(1, 1)))

    
    io = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)
    io = cv2.resize(io, (0, 0), fx=args.resize, fy=args.resize)

    io_shape_orig = np.array(io.shape)
    
    #add half the stride as padding around the image, so that we can crop it away later
    io = np.pad(io, [(stride_size//2, stride_size//2), (stride_size//2, stride_size//2), (0, 0)], mode="reflect")
    
    io_shape_wpad = np.array(io.shape)
    
    #pad to match an exact multiple of unet patch size, otherwise last row/column are lost
    npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
    npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

    io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

    arr_out = sklearn.feature_extraction.image.extract_patches(io,(patch_size,patch_size,3),stride_size)
    arr_out_shape = arr_out.shape
    arr_out = arr_out.reshape(-1,patch_size,patch_size,3)

    #in case we have a large network, lets cut the list of tiles into batches
    output = np.zeros((0,checkpoint["num_classes"]))
    for batch_arr in divide_batch(arr_out,batch_size):
        
        arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

        # ---- get results
        output_batch = model(arr_out_gpu)

        # --- pull from GPU and append to rest of output 
        output_batch = output_batch.detach().cpu().numpy()
        output = np.append(output,output_batch,axis=0)


    tileclass = np.argmax(output, axis=1)
    predc,predccounts=np.unique(tileclass, return_counts=True)
    for c,cc in zip(predc,predccounts):
        print(f"class/count: \t{c}\t{cc}")

    print(f"predicted class:\t{predc[np.argmax(predccounts)]}")

