python run_infer.py \
--gpu=0 \
--batch_size=64 \
--model_mode=fast \
--model_path=logs/monuseg/256x256_164x164 \
tile \
--input_dir=/data/yuxinyi/MoNuSeg/test/images \
--output_dir=results/monuseg
