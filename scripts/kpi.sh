# Đọc mô tả scripts sau 

# train
python -u run_main.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/kpi_processed/ --data_path EnodebF121.csv --model_id KPI_F121_96_96 --model_comment 'KPI_GPT2' --model TimeLLM --data TelecomKPI --features M --seq_len 96 --label_len 48 --pred_len 96 --enc_in 5 --c_out 5 --des 'Exp_KPI_F121' --train_epochs 10 --batch_size 8 --learning_rate 0.0005 --llm_layers 6 --llm_model GPT2 --llm_dim 768 --use_amp --target ps_traffic_mb
#test ( đổi is_training = 0 )
python -u run_main.py --task_name long_term_forecast --is_training 0 --root_path ./dataset/kpi_processed/ --data_path EnodebF121.csv --model_id KPI_F121_96_96 --model_comment 'KPI_GPT2' --model TimeLLM --data TelecomKPI --features M --seq_len 96 --label_len 48 --pred_len 96 --enc_in 5 --c_out 5 --des 'Exp_KPI_F121' --train_epochs 10 --batch_size 8 --learning_rate 0.0005 --llm_layers 6 --llm_model GPT2 --llm_dim 768 --use_amp --target ps_traffic_mb