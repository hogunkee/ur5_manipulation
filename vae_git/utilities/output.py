import numpy as np
import os
import imageio
import torch, torchvision

to_np = lambda x: x.data.cpu().numpy()
background_img = imageio.imread(os.path.join('../ur5_mujoco', 'background.png')).astype(np.float32).transpose([2,0,1])/255

def write_logger(config, logger):
	np.savetxt("logger.dat", np.vstack((np.arange(0, config.num_epochs, config.num_every_nth_epoch), logger["loss"])).T, delimiter=",", header="epoch,loss", fmt=["%i", "%2.8e"], comments="")

def write_observations(config, epoch, z_fixed, v, logger, loss):
	# store samples ..
        if config.remove_bg:
            samples = v.Decoder(v.mu)[:64] + torch.tensor(background_img).float().cuda()
        else:
            samples = v.Decoder(v.mu)[:64]
	# samples = v.Decoder(z_fixed)
        torchvision.utils.save_image(samples.data, '%s/%03d.png' % (config.img_path, epoch / config.num_every_nth_epoch), normalize=True)

        # .. checkpoint ..
        torch.save(v.state_dict(), os.path.join(config.ckpt_path, 'v.pth'))

        # .. and collect loss
        logger["loss"] = np.append(logger["loss"], to_np(loss))

