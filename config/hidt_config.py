"""
Lambdas order:
1. loss_adv + loss_adv_r
2. loss_rec + loss_rec_r + loss_cyc
3. loss_c + loss_c_r
4. loss_s
5. loss_s_r
6. loss_dist
"""

PROFILE = "BULAT"  # NASTIA, SAIDASH, SPHERE

config = {
    "learning_rate": 3e-4,
    "lambdas": [5, 2, 1, 0.1, 4, 1],
    "n_epochs": 100,
    "batch_size": 2,
    "num_workers": 0,
    "save_period": 800,
    "log_frequency": 100,
    "picture_frequency": 100,
    "device": 'cuda:4',
    "data_path": "data/images",
    "send_wandb": 150,
    "from_pretrained": False,
    "checkpoint_path": "notcheckpoints/step=30400.pt",
    "save_checkpoint_path": "notcheckpoints/",
    "debug": True,
    "test_size": 0.05,
    "validate_period": 500,
    "validate_start": 1000,
}

if PROFILE == "BULAT":
    config["data_path"] = "/data/bvshelhonov/images"
    config["save_checkpoint_path"] = "notcheckpoints/"
elif PROFILE == "NASTIA":
    config["data_path"] = "/data/bvshelhonov/images"
    config["save_checkpoint_path"] = "notcheckpoints/"
    raise NotImplementedError()
elif PROFILE == "SAIDASH":
    config["data_path"] = "/data/bvshelhonov/images"
    config["save_checkpoint_path"] = "notcheckpoints/"
    raise NotImplementedError()
elif PROFILE == "SPHERE":
    config["data_path"] = "/data/bvshelhonov/images"
    config["save_checkpoint_path"] = "notcheckpoints/"
    raise NotImplementedError()
else:
    raise ValueError("Invalid profile")
