import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--object", default='car', help='Mention object for Object specific GAN training')
parser.add_argument("--predict", action='store_true')
parser.add_argument("--name",  type=str, default="enlightening", help="For training, the name of the directory where checkpoints are created, for testing directory from which it is loaded")
parser.add_argument("--gpu_ids",  type=str, default="0", help="GPU ids to use, to use 2 GPUs set to 0,1")
parser.add_argument("--data_path",  type=str, default="/home/vq218944/Downloads/EnlightenGAN_Data", help="Path to the downloaded dataset")
parser.add_argument("--loss_type",  type=str, default="stylefeat", help="Choose between relu5_1 | relu5_3 | stylefeat")

opt = parser.parse_args()

if opt.train:
	os.system("python train.py \
		--dataroot {} \
		--no_dropout \
		--name {} \
		--model single \
		--dataset_mode unaligned \
		--which_model_netG sid_unet_resize \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
        --patchD_3 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
		--fineSize 256 \
        --patchSize 32 \
		--skip 1 \
		--batchSize 2 \
        --self_attention \
		--use_norm 1 \
		--use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
		--instance_norm 0 \
		--vgg 1 \
        --vgg_choose {} \
		--gpu_ids 0 \
		--resize_or_crop resize_and_crop \
		--object {} \
		--display_port={}".format(opt.data_path, opt.name, opt.loss_type, opt.object, opt.port))

elif opt.predict:
	for i in range(1, 2):
	        os.system("python predict.py \
	        	--dataroot {} \
	        	--name {} \
	        	--model single \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
                --self_attention \
                --times_residual \
				--gpu_ids {} \
	        	--instance_norm 0 --resize_or_crop='no'\
	        	--which_epoch latest".format(opt.data_path, opt.name, opt.gpu_ids))
