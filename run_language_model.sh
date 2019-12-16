#Ignore the TRAIN and TEST files I define below. 

#   1. Fill in load_fulltext() on line 67. This is called on line 549.
#   2. Find a trunation length for the fulltexts (same problem as with article classification)
 

export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw


#Test ------------------------
python arxiv_lm_finetuning.py \
    --output_dir=output/language-model-title \
    --overwrite_output_dir \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --save_steps=100 \
    --save_total_limit=5 \
    --N=100 \
    --type='title' \
    --per_gpu_train_batch_size=2 \
    --num_train_epochs=1


python arxiv_lm_finetuning.py \
    --output_dir=output/language-model-abstract \
    --overwrite_output_dir \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --save_steps=100 \
    --save_total_limit=5 \
    --N=100 \
    --type='abstract' \
    --per_gpu_train_batch_size=2 \
    --num_train_epochs=1


#Main -------------------------
#python arxiv_lm_finetuning.py \
#    --output_dir=output/language-model-title \
#    --overwrite_output_dir \
#    --model_type=gpt2 \
#    --model_name_or_path=gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --save_steps=100000000 \
#    --save_total_limit=5\
#    --N=10000000 \
#    --type='title' \
#    --per_gpu_train_batch_size=2 \
#    --num_train_epochs=10 

#python arxiv_lm_finetuning.py \
#    --output_dir=output/language-model-abstract \
#    --overwrite_output_dir \
#    --model_type=gpt2 \
#    --model_name_or_path=gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --save_steps=10000000000 \
#    --N=10000000 \
#    --type='abstract' \
#    --per_gpu_train_batch_size=2

#python arxiv_lm_finetuning.py \
#    --output_dir=output/language-model-fulltext \
#    --overwrite_output_dir \
#    --model_type=gpt2 \
#    --model_name_or_path=gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --save_steps=10000000000 \
#    --N=10000000 \
#    --type='fulltext'


