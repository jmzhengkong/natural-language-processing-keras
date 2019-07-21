model=cnn
python3 train.py \
--hparam=$model.json \
--batch_size=128 \
--epoch=30 \
--monitor=val_loss \
--gpu=1 \
--model_dir=$model \
--model=$model \
--data_dir=hateSpeech.40.npz

model=rnn
python3 train.py \
--hparam=$model.json \
--batch_size=128 \
--epoch=10 \
--monitor=val_loss \
--gpu=2 \
--model_dir=$model \
--model=$model \
--data_dir=hateSpeech.40.npz \
