# import os
# import shutil

# ct_path = "/root/autodl-tmp/segmentation/MASS/Dataset/BTCV/BTCV_256_256_32"
# mr_path = "/root/autodl-tmp/segmentation/MASS/Dataset/CHAOS/CHAOS_256_256_32"

# dirs = os.listdir(ct_path)
# for dir_ in dirs:
#     if dir_ == "train":
#         files = os.listdir(os.path.join(ct_path, dir_, "imagesTr"))
#         for file in files:
#             ct_name = file.split(".")[0].split("g")[-1]
#             if not os.path.exists(f"/root/autodl-tmp/segmentation/MASS/Dataset/CT_MR/train/CT_{int(ct_name)}"):
#                 shutil.copytree(os.path.join(ct_path, dir_, "imagesTr", file), f"/root/autodl-tmp/segmentation/MASS/Dataset/CT_MR/train/CT_{int(ct_name)}")
#     else:
#         files = os.listdir(os.path.join(ct_path, dir_, "imagesTr"))
#         for file in files:
#             ct_name = file.split(".")[0].split("g")[-1]
#             if not os.path.exists(f"/root/autodl-tmp/segmentation/MASS/Dataset/CT_MR/valid/CT_{int(ct_name)}"):
#                 shutil.copytree(os.path.join(ct_path, dir_, "imagesTr", file), f"/root/autodl-tmp/segmentation/MASS/Dataset/CT_MR/valid/CT_{int(ct_name)}")
            
# dirs = os.listdir(mr_path)
# for dir_ in dirs:
#     if dir_ == "train":
#         files = os.listdir(os.path.join(mr_path, dir_))
#         for file in files:
#             mr_name = int(file)
#             if not os.path.exists(f"/root/autodl-tmp/segmentation/MASS/Dataset/CT_MR/train/MR_{mr_name}"):
#                 shutil.copytree(os.path.join(mr_path, dir_, file), f"/root/autodl-tmp/segmentation/MASS/Dataset/CT_MR/train/MR_{mr_name}")
#     else:
#         files = os.listdir(os.path.join(mr_path, dir_))
#         for file in files:
#             mr_name = int(file)
#             if not os.path.exists(f"/root/autodl-tmp/segmentation/MASS/Dataset/CT_MR/valid/MR_{mr_name}"):
#                 shutil.copytree(os.path.join(mr_path, dir_, file), f"/root/autodl-tmp/segmentation/MASS/Dataset/CT_MR/valid/MR_{mr_name}")

# import os
import shutil
import os
files = os.listdir("/root/autodl-tmp/abdomen/labelsTr")
for file in files:
    shutil.copy(os.path.join("/root/autodl-tmp/abdomen/labelsTr", file), f"/root/autodl-tmp/segmentation/MASS/Dataset/BTCV/data/label_{int(file.split('.')[0].split('l')[-1])}.nii.gz")