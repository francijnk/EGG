python3 -m ancm.train --data_path ancm/data/input_data/img-5-1.npz \
 --vocab_size 16 --n_epochs 30 --max_len 5 --length_cost 0.01 --sender_cell lstm \
 --receiver_cell lstm --batch_size 32 --sender_hidden 50 --receiver_hidden 50 \
 --sender_embedding 10 --receiver_embedding 10 --sender_lr 1e-2 --receiver_lr 2e-3 \
 --sender_entropy_coeff 0.01 --receiver_entropy_coeff 0.001 --evaluate --results_folder runs/ \
 --filename baseline --random_seed 42 --image