TASK_NAME='custom'
TEACHER_MODEL='../model_hub/roberta_base_model_test//'
STUDENT_MODEL='../model_hub/roberta_base_model_test//'
GLUE_DIR=/zhangleisx4614/code/pytorch_bert_chinese_classification-main/data/

python -m torch.distributed.launch run_glue_distillation.py \
    --model_type bert \
    --teacher_model $TEACHER_MODEL \
    --student_model $STUDENT_MODEL \
    --task_name $TASK_NAME \
    --num_hidden_layers 3 \
    --alpha 0.5 \
    --evaluate_during_training \
    --beta 100.0 \
    --local_rank 0 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 4.0 \
    --output_dir ./tmp/$TASK_NAME/