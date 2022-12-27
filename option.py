import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser("Train parser")

    def initialize(self):
        parser = self.parser

        parser.add_argument('--adj_thresh', type=float, default=0.45, help="threshold of adj edges")
        parser.add_argument('--max_epochs', type=int, default=150, help='max epochs')
        parser.add_argument('--DATA_DIR', type=str, default='data/correlation', help='adj dir')
        parser.add_argument('--LABLE_DIR', type=str, default='data/class_label', help='label dir')
        parser.add_argument('--CSV_FILE', type=str, default='data/Phenotypic_V1_0b_preprocessed1.csv', help='csv file')
        parser.add_argument('--BATCH_SIZE', type=int, default=32, help='batch_size')
        parser.add_argument('--class_nums', type=int, default=16, help='the number of multi-source sites')
        parser.add_argument('--start_epoch', type=int, default=120, help='start from which epoch')
        parser.add_argument('--continue_fold', type=int, default=0, help='start from which fold')
        parser.add_argument('--fold', type=int, default=5, help='5-FOLD-CV')
        parser.add_argument('--seed', type=int, default=1, help='random seed')
        parser.add_argument('--model_dir', type=str, default='./models', help='model dir')
        parser.add_argument('--ROIs', type=float, default=200, help='brain regions cc200')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--workers', type=int, default=4, help='num_workers')
        parser.add_argument('--visual_path', type=str, default='./visuals', help='visual train graph')

        # M models' layers opts
        parser.add_argument('--in_dim', type=int, default=8,help='model hypernode dimental')
        parser.add_argument('--hidden_dim', type=float, default=8,help='model hypernode dimental')
        parser.add_argument('--out_dim', type=float, default=2,help='classification labels')

        # G and D models' layers opts
        parser.add_argument('--hidden1', type=int, default=32)
        parser.add_argument('--hidden2', type=int, default=64)
        parser.add_argument('--hidden3', type=int, default=32)
        parser.add_argument('--hidden4', type=int, default=8)
        parser.add_argument('--dropout', type=int, default=0.5)

        #train model opts
        parser.add_argument('--lr_G', type=float, default=1e-4, help='learning rate for G')
        parser.add_argument('--lr_D', type=float, default=1e-5, help='learning rate for D')
        parser.add_argument('--lr_M', type=float, default=1e-4, help='learning rate for M')
        parser.add_argument('--momentum', type=float, default=0.9,help='momentum')
        parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
        parser.add_argument('--iter_D', type=int, default=5, help='iter D for')

        opt = parser.parse_args()
        return opt
