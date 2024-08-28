# Self-Supervised-Leaf-Segmentation

## Introduction

​	这是根据https://github.com/lxfhfut/Self-Supervised-Leaf-Segmentation修改后的代码，主要功能是分割出叶片，构建真值。modify_RGB_num164是叶片原始图像，modify_TRUTH_num164是与此相对应的真值。

## Getting Started

### 一、分割叶片构建真值

```
python leaf_segmneter.py
```

其中修改的主要部分如下

```python
# threshold = 0.4
labels = measure.label(image_labels)
mean_img = mean_image(rgb_image, labels)
absolute_greenness, relative_greenness = cal_greenness(mean_img)
greenness = np.multiply(relative_greenness, (absolute_greenness > args.at).astype(np.float64))
thresholded = 255 * ((greenness > args.rt).astype("uint8"))
relative_greenness_gray = copy.deepcopy(relative_greenness) #深拷贝
relative_greenness[relative_greenness > threshold] = 1 #灰度图二值化
relative_greenness[relative_greenness <= threshold] = 0 #灰度图二值化
relative_greenness_color = np.expand_dims(relative_greenness, axis=-1).astype(np.uint8) # 扩展为三通道
combine = rgb_image // 2 + relative_greenness_color // 2
save_result_path = os.path.join(args.save_path, f'wtruth_{i}.png')
save_result_img(save_result_path, rgb_image, labels, combine,absolute_greenness, relative_greenness_gray, thresholded)

cv2.imwrite(save_result_path, relative_greenness)
```

### 二、检查彩色叶子图像与对应的二值图（叶子真值）能否重合

我们在构建出原始图像的真值后后需要检查彩色叶子图像与对应的二值图（叶子真值）能否重合，人工观察将差异较大的删除，代码如下

```python
import cv2
import glob
import numpy as np

path1 = r'/home/xplv/fenghao/Self-Supervised-Leaf-Segmentation/modify_Input_num164/img_*.jpg'
files1 = glob.iglob(path1)
sorted_files1 = sorted(files1)
path2 = r'/home/xplv/fenghao/Self-Supervised-Leaf-Segmentation/modify_Output_num164/truth_*.png'
files2 = glob.iglob(path2)
sorted_files2 = sorted(files2)

for i in range(0, 163):
    print(sorted_files1[i])
    print(sorted_files2[i])
    img1 = cv2.imread(sorted_files1[i], -1)  # img1 is rgb image
    img2 = cv2.imread(sorted_files2[i], -1) * 255  # img2 is grey image
    img2 = np.expand_dims(img2, axis=-1).astype(np.uint8) # expand dim
    combine = img1 // 2 + img2 // 2
    cv2.imwrite(f'Output_163/combine_{i}.png', combine)
```

