python3 ancm/train.py --data_path "ancm/data/input_data/[2, 2, 2, 2, 2, 2, 2, 2]_4_distractors.npz" \
  --vocab_size 16 --n_epochs 5 --max_len 3 --length_cost 0.001 \
  --sender_cell lstm --receiver_cell lstm \
  --batch_size 32 \
  --channel erasure --error_prob 0.1 \
  --sender_hidden 50 --receiver_hidden 50 --sender_embedding 10 --receiver_embedding 10 \
  --sender_lr 1e-3 --receiver_lr 2e-4 --sender_entropy_coeff 0.01 --receiver_entropy_coeff 0.001 \
  --evaluate --results_folder runs/ --filename baseline --random_seed 42 \
  --wandb_project cezary_test --wandb_entity koala-lab 
