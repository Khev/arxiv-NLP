#Title
python arxiv_lm_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=output/language-model-title \
    --prompt_length=3 \
    --num_docs=10 \
    --length=20

#Abstracts
python arxiv_lm_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=output/language-model-abstract \
    --prompt_length=10 \
    --num_docs=10 \
    --length=150 
