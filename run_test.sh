python3 ancm/train.py --perceptual_dimensions "[2] * 8" --n_distractors 4 --data_seed 42 \
  --vocab_size 16 --n_epochs 5 --max_len 5 --length_cost 0.001 \
  --sender_cell lstm --receiver_cell lstm \
  --batch_size 32 \
  --channel symmetric --error_prob 0.1 \
  --sender_hidden 50 --receiver_hidden 50 --sender_embedding 10 --receiver_embedding 10 \
  --sender_lr 1e-3 --receiver_lr 2e-4 --sender_entropy_coeff 0.01 --receiver_entropy_coeff 0.001 \
  --mode rf --evaluate  --dump_data_folder ancm/data/input_data/ --dump_results_folder runs/ --filename baseline
