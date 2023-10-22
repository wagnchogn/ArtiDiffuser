export OPENAI_LOGDIR=""
export DATA_DIR=''

MODEL_FLAGS="--image_size 256 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4096 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 1"
python scripts/image_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
