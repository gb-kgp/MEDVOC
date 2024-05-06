To run the run_summarization.py borrowed from huggingface implemenation, we introduce two custom parameters:
  1. tokenizer_type which could be {BART/PEGASUS}
  2. domain_token_path which is the path of updated model vocabulary.

We present an example of using the updated python  script as follows:
```
python run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --train_file #path to PAC Train \
    --validation_file #path to PAC Valid csv \
    --output_dir #path where you wish to store data \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --eval_steps 2500 \
    --num_train_epochs 5 \
    --save_strategy steps \
    --save_steps 2500 \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --gradient_accumulation_steps 2 \
    --tokenizer_type #BART or PEGASUS \
    --domain_token_path #path_to_vocabulary_updated_tokenizer  \
    --max_source_length 700 \
    --max_target_length 200 \
    --text_column input_text \
    --summary_column target_text

```
