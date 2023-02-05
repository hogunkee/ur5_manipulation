import os
import imageio
from PIL import Image

path = [f"./capture/{i}" for i in os.listdir("./capture")]
path.sort()
paths = [ Image.open(i) for i in path]
imageio.mimsave('./test.gif', paths, fps=4)
