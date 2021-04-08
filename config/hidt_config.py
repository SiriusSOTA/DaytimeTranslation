"""
Lambdas order:
1. loss_adv + loss_adv_r
2. loss_rec + loss_rec_r + loss_cyc
3. loss_c + loss_c_r
4. loss_s
5. loss_s_r
6. loss_dist
"""

config = {
    "learning_rate": 3e-4,
    "lambdas": [5, 2, 1, 0.1, 4, 1],
    "n_epochs": 100,
    "batch_size": 4,
    "save_period": 800,
    "log_frequency": 100,
    "picture_frequency": 100,
    "device": 'cuda:3',
    "data_path": "data/images",
    "send_wandb": 150,
}
