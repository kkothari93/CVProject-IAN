import argparse
import os
import time
import glob
import numpy as np

parser = argparse.ArgumentParser(description='Template for adding Arguments')
parser.add_argument('-d','--records_loc', metavar='ARCH', default='./data/celebA_tfrecords/celeba*',
                    help='Location of records')
parser.add_argument('-r', '--results_dir', default='',
                    help='Output Directory')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-l', '--lat_size', default=100, type=int,
                    metavar='N', help='Latent Vector size (default: 100)')
parser.add_argument('-s', '--ndirs_swd', default=10000, type=int,
                    metavar='N', help='SWD Ndirs (default: 10000)')
parser.add_argument('-a', '--w_adv', default=1, type=int)
parser.add_argument('-i', '--w_img', default=3, type=int)
parser.add_argument('-f', '--w_feat', default=1, type=int)
parser.add_argument('-k', '--w_kl', default=1, type=int)
parser.add_argument('-g', '--w_gp', default=10, type=int)


def get_kwargs():
	return vars(parser.parse_args())

	
		

	
		
#if __name__ == '__main__':
#	main()