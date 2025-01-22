python3 ancm/train.py --data_path ancm/data/input_data/visa-4-500.npz \
  --channel erasure --error_prob 0.1 \
  --vocab_size 10 --n_epochs 20 --max_len 5 --length_cost 0.01 \
  --sender_cell lstm --receiver_cell lstm \
  --batch_size 32 \
  --sender_hidden 50 --receiver_hidden 50 --sender_embedding 12 --receiver_embedding 12 \
  --sender_lr 1e-3 --receiver_lr 5e-4 \
  --sender_entropy_coeff 0.01 --receiver_entropy_coeff 0.001 \
  --evaluate --results_folder runs/ --filename baseline --random_seed 50
