import os
import sys

file_path = os.path.abspath(os.getcwd())
sys.path.append(os.path.join(file_path, '../..', 'UnseenObjectClustering'))

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class UCNModule():
    def __init__(self):
        self.pretrained = 'data/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth'
        self.cfg_file = 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml'
        cfg_from_file(self.cfg_file)

        self.network_name = 'seg_resnet34_8s_embedding'
        self.network_data = torch.load(self.pretrained)
        self.network = networks.__dict__[self.network_name](2, 64, self.network_data).type(dtype)
        self.network.eval()

        self.network_crop = None
        self.target_resolution = 96

    def get_masks(self, image, data_format='HWC'):
        im_tensor = torch.from_numpy(image).to(torch.float32)
        if data_format=='HWC':
            im_tensor = im_tensor.permute(2, 0, 1)
        im_tensor = im_tensor.unsqueeze(0).type(dtype)

        features = self.network(image, None).cpu().detach()
        out_label, selected_pixels = clustering_features(features, num_seeds=100)
