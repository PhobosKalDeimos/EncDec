import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--lstm_layers", type=int, default=4, help="_")
    parser.add_argument("--latent_size", type=int, default=80, help="_")
    parser.add_argument("--window_size", type=int, default=5, help="_")
    parser.add_argument("--prediction_length", type=int, default=5, help="_")
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=3, help="_")
    parser.add_argument("--lr", type=float, default=0.001, help="_")
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--frequency", type=int, default=10, help="_")
    parser.add_argument("--train_split", type=int, default=0.9, help="_")
    parser.add_argument("--test_split_value", type=int, default=0.8, help="_")
    parser.add_argument("--test_split",  action='store_true',
                        help="split test set in two subset")
    # Paths parameters
    parser.add_argument("--dataset_folder", type=str, default=None,
                        help="path of the folder with train/val/test sets")
    parser.add_argument("--resume",  action='store_true',
                        help="resume model for test")
    parser.add_argument("--model_path", type=str, default="checkpoints/model_checkpoint.pth",
                        help="path of the folder of the model to resume")
    
    args = parser.parse_args()
    
    if args.dataset_folder is None:
        raise Exception("You should set parameter --dataset_folder")
    
    if not os.path.exists(args.dataset_folder):
        raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")
    
    return args