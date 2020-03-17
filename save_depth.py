from loaders import *

# path = r'/media/SENSETIME\qiudi/Data/relighting_download/Xu_data/'
path = r'/mnt/lustre/qiudi/lighting_data/Xu_data/'
dataloader = XuLoader(path, 'train')

for i in range(len(dataloader)):
    
    print(i)
    view = dataloader.files['train'][i]
    print(view)
    # dataloader._save_npy(pjoin(dataloader.root, 'orgTrainingImages', view))
    dataloader._save_depth(view)
