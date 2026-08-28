import argparse
parser = argparse.ArgumentParser(description='parameters')

parser.add_argument('--mode', type=str,default='rotation_train_resnet101', help='different proxy task and model')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--train_path', type=str, default='./Dataset/generative_image/train')
parser.add_argument('--test_path', type=str, default='./Dataset/generative_image/test')

parser.add_argument('--downstream_train_data', type=str, default='./Dataset/true_image/train/images')
parser.add_argument('--downstream_train_label', type=str, default='./Dataset/true_image/train/masks')

parser.add_argument('--downstream_test_data', type=str, default='./Dataset/true_image/test/images')
parser.add_argument('--downstream_test_label', type=str, default='./Dataset/true_image/test/masks')
parser.add_argument('--checkpoint', type=str, default=None, help='Optional pretrained checkpoint path for legacy downstream modes')
parser.add_argument('--device', type=str, default=None, help='Override device, e.g. cuda:0 or cpu')
args, unknown = parser.parse_known_args()
