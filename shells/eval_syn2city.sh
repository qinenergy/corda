CUDA_VISIBLE_DEVICES=0 python3 evaluateUDA.py --full-resolution -m deeplabv2_synthia --model-path ./checkpoint/synthia/checkpoint-iter250000.pth
# Note that Synthia mIoU result should be multiplied by *19/16 because of the missing classes.
