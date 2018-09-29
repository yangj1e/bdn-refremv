#!/usr/bin/env bash
python ./test.py --dataroot ./samples \
    --batchSize 1 \
    --norm batch \
    --which_model_netG cascade_unet \
    --ns 7,5,5 \
    --iteration 1 \
    --outf ./output \
    --netG ./model/model.pth \
    --real