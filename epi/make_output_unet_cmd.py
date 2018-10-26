#v1
#26/10/2018

import argparse
import os
import glob
import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided
import torch
from unet import UNet

# ----- parse command line arguments
parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",
                    nargs="*")

parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
parser.add_argument('-o', '--outdir', help="outputdir, default ./output/", default="./output/", type=str)
parser.add_argument('-r', '--resize', help="resize factor 1=1x, 2=2x, .5 = .5x", default=1, type=float)
parser.add_argument('-m', '--model', help="model", default="best_model.ph", type=str)
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

patch_size = args.patchsize
block_shape = np.array((patch_size, patch_size, 3))

# ----- load network
device = torch.device(args.gpuid if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(args.model)
model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
             padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
             up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
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
    print(f"saving to : \t {newfname_class}")

    if not args.force and os.path.exists(newfname_class):
        print("Skipping as output file exists")
        continue

    cv2.imwrite(newfname_class, np.zeros(shape=(1, 1)))

    io = cv2.imread(fname)
    io = cv2.resize(io, (0, 0), fx=args.resize, fy=args.resize)

    io_shape_orig = np.array(io.shape)


    #pad to match a multiple of unet patch size
    npad0 = int(np.ceil(io_shape_orig[0] / patch_size) * patch_size - io_shape_orig[0])
    npad1 = int(np.ceil(io_shape_orig[1] / patch_size) * patch_size - io_shape_orig[1])

    io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

    if not io.flags.contiguous:
        io = np.ascontiguousarray(io)

    io_shape = io.shape

    # ---reshape image to batch sizes
    new_shape = tuple(io_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(io.strides * block_shape) + io.strides

    arr_out = as_strided(io, shape=new_shape, strides=new_strides)
    arr_out = arr_out.reshape([-1, patch_size, patch_size, arr_out.shape[-1]])

    arr_out_gpu = torch.from_numpy(arr_out.transpose(0, 3, 1, 2) / 255).cuda(device).type('torch.cuda.FloatTensor')

    # ---- get results
    output = model(arr_out_gpu)

    # --- reshape results
    output = output.detach().squeeze().cpu().numpy()
    output = output.transpose((0, 2, 3, 1))
    output = output.reshape(new_shape[0:-1] + (-1,))

    output = output.squeeze().swapaxes(1, 2).reshape(-1, output.shape[1] * output.shape[3], output.shape[-1])

    output = output[0:io_shape_orig[0], 0:io_shape_orig[1], :] #remove paddind, crop back

    # --- save output

    # cv2.imwrite(newfname_class, (output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1)).astype(np.uint8))
    cv2.imwrite(newfname_class, output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1))
