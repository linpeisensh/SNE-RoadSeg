import os
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2labelim, tensor2confidencemap
from models.sne_model import SNE
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2 as cv


class dataset():
	def __init__(self):
		self.num_labels = 2

def load_images(path_to_sequence):
	return [os.path.join(path_to_sequence, img) for img in os.listdir(path_to_sequence)]

def get_cam_param(ins):
	if ins < 3:
		fx = 718.856
		ox = 607.1928
		oy = 185.2157
		bf = 387.5744
	elif ins == 3:
		fx = 721.5377
		ox = 609.5593
		oy = 172.854
		bf = 387.5744
	else:
		fx = 707.0912
		ox = 601.8873
		oy = 183.1104
		bf = 379.8145
	camParam = torch.tensor([[fx, 0.000000e+00, ox],
							 [0.000000e+00, fx, oy],
							 [0.000000e+00, 0.000000e+00, 1.000000e+00]], dtype=torch.float32)  # camera parameters
	return camParam, bf


if __name__ == '__main__':
	opt = TestOptions().parse()
	opt.num_threads = 1
	opt.batch_size = 1
	opt.serial_batches = True  # no shuffle
	opt.isTrain = False

	example_dataset = dataset()
	model = create_model(opt, example_dataset)
	model.setup(opt)
	model.eval()

	file_path = os.path.join(opt.data_path, opt.sequence, 'image_2')
	left_filenames = load_images(file_path)

	depth_path = os.path.join(opt.data_path, opt.sequence)
	depth_filenames = load_images(file_path)

	save_path = os.path.join(opt.save_path, opt.sequence)

	camParam, bf = get_cam_param(int(opt.sequence))
	print(left_filenames[0])
	print(depth_filenames[0])
	print(save_path)
	# if you want to use your own data, please modify rgb_image, depth_image, camParam and use_size correspondingly.
	for i in range(len(left_filenames)):
		rgb_image = cv.cvtColor(cv.imread(left_filenames[i]), cv.COLOR_BGR2RGB)
		depth_image = cv.imread(depth_filenames[i], cv.IMREAD_ANYDEPTH)
		depth_image = depth_image.astype(np.float32)
		row = depth_image.shape[0]
		col = depth_image.shape[1]
		depth_image = np.true_divide(np.ones((row, col)), depth_image)
		depth_image = depth_image * bf * 256.0
		oriHeight, oriWidth, _ = rgb_image.shape
		oriSize = (oriWidth, oriHeight)

		# resize image to enable sizes divide 32
		use_size = (1248, 384)
		rgb_image = cv.resize(rgb_image, use_size)
		rgb_image = rgb_image.astype(np.float32) / 255

		# compute normal using SNE
		sne_model = SNE()

		normal = sne_model(torch.tensor(depth_image.astype(np.float32) / 1000), camParam)
		normal_image = normal.cpu().numpy()
		normal_image = np.transpose(normal_image, [1, 2, 0])
		# cv.imwrite(os.path.join('examples', 'normal.png'), cv.cvtColor(255*(1+normal_image)/2, cv.COLOR_RGB2BGR))
		normal_image = cv.resize(normal_image, use_size)

		rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(dim=0)
		normal_image = transforms.ToTensor()(normal_image).unsqueeze(dim=0)

		with torch.no_grad():
			pred = model.netRoadSeg(rgb_image, normal_image)

			palet_file = 'datasets/palette.txt'
			impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3 * 256))
			pred_img = tensor2labelim(pred, impalette)
			pred_img = cv.resize(pred_img, oriSize)
			prob_map = tensor2confidencemap(pred)
			prob_map = cv.resize(prob_map, oriSize)
			# cv.imwrite(os.path.join('examples', 'pred.png'), pred_img)
			cv.imwrite(os.path.join(save_path, '{0:06}.png'.format(i)), prob_map)
			print('{} frames'.format(i))
