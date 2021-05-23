cp ./extra/trainUDA_synthia_simpleaux.py ./
CUDA_VISIBLE_DEVICES=0 python3 -u trainUDA_synthia_simpleaux.py --config ./configs/configUDA_syn2citystereo.json --name UDA_synthia_stereo_simpleaux | tee ./synthia_stereo_simpleaux.log
