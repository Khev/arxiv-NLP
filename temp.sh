input='output/language-model-title/test-docs-for-prompts.txt'
while IFS= read -r line
do
  python run_generation.py --model_type=gpt2 --model_name_or_path=output/language-model-title --length=50 --prompt="$line"
done < "$input"
