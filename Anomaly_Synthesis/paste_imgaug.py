import numpy as np
import os
from PIL import Image
import os
import random
from skimage.measure import label
from multiprocessing import Pool
import imgaug.augmenters as iaa

def augment(augs, segmask):
    Lim_patch = segmask.convert('L')
    mask = Lim_patch.point(table,'L')
    mask = mask.convert('RGB')
    segmask_augmented = augs(image=np.asarray(segmask))
    segmask_augmented = Image.fromarray(segmask_augmented)
    mask_augmented = augs(image=np.asarray(mask))
    mask_augmented = Image.fromarray(mask_augmented)
    mask_augmented = mask_augmented.convert('L')
    mask = Image.fromarray((np.asarray(mask_augmented)>180)*255.0)#去插值产生的黑边
    mask = mask.convert('1')
    return segmask_augmented, mask

def deformation(segmask, dis): #翻转，旋转，形变，缩放，colorjitter 等等
    aug = iaa.Sequential(augmenters)
    augmenters_det = aug.to_deterministic()#固定本循环中的增强策略
    segmask_augmented, mask = augment(augmenters_det, segmask)
    segmask_augmenteds = [segmask_augmented]
    masks = [mask]
    pos = ran_pos(dis)
    to_location_h = int(pos[0] - segmask_augmented.size[0]/2)
    to_location_w = int(pos[1] - segmask_augmented.size[1]/2)
    location = (to_location_w, to_location_h)
    return segmask_augmenteds, masks, location

def ran_pos(dis): #保证粘贴的位置距离中心不太远，防止贴到眼睛外面
    pos = [random.randint(1, img_size), random.randint(1, img_size)]
    while (pos[0]-img_size/2)**2+(pos[1]-img_size/2)**2>(img_size/2-dis)**2:
        pos = [random.randint(1, img_size), random.randint(1, img_size)]
    return pos

def past(pasted_imgs, img_masks, past_patchs, patch_masks, location):
    for aug_num in range(len(pasted_imgs)):
        pasted_imgs[aug_num].paste(past_patchs[aug_num], location, mask=patch_masks[aug_num])
        img_masks[aug_num].paste(past_patchs[aug_num], location, mask=patch_masks[aug_num])
    return pasted_imgs, img_masks

def past_MA(pasted_imgs, i, img_masks):#加重采样
    random.seed(i+1)
    np.random.seed(i+1)
    list_MA = range(1, num_MA)
    num = random.randint(3, int(num_MA*1.5)) #随机采样次数，可以大于总数重复采样，也可以固定一个随机范围
    num_MA_rand = np.random.choice(list_MA, num)+1
    for mask_MA in num_MA_rand:
        single_mask = Image.fromarray((labeled_MA==mask_MA)*1.0)
        single_bbox = single_mask.getbbox()
        MA_patch = MA.crop(single_bbox)#得到病灶的patch
        MA_patchs, patch_masks, location = deformation(MA_patch, 10)#用patch_mask, location来得到完整的mask
        pasted_imgs, img_masks = past(pasted_imgs, img_masks, MA_patchs, patch_masks, location)
    return pasted_imgs, img_masks

def past_HE(pasted_imgs, i, img_masks):
    random.seed(i+2)
    np.random.seed(i+2)
    list_HE = range(1, num_HE)
    num = random.randint(1, int(num_HE*1.5))
    num_HE_rand = np.random.choice(list_HE, num)+1
    for mask_HE in num_HE_rand:
        single_mask = Image.fromarray((labeled_HE==mask_HE)*1.0)
        single_bbox = single_mask.getbbox()
        HE_patch = HE.crop(single_bbox)
        HE_patchs, patch_masks, location = deformation(HE_patch, 20)
        pasted_imgs, img_masks = past(pasted_imgs, img_masks, HE_patchs, patch_masks, location)
    return pasted_imgs, img_masks

def past_EX(pasted_imgs, i, img_masks):
    random.seed(i+3)
    np.random.seed(i+3)
    list_EX = range(1, num_EX)
    num = random.randint(1, int(num_EX*1.5))
    num_EX_rand = np.random.choice(list_EX, num)+1
    for mask_EX in num_EX_rand:
        single_mask = Image.fromarray((labeled_EX==mask_EX)*1.0)
        single_bbox = single_mask.getbbox()
        EX_patch = EX.crop(single_bbox)
        EX_patchs, patch_masks, location = deformation(EX_patch, 20)
        pasted_imgs, img_masks = past(pasted_imgs, img_masks, EX_patchs, patch_masks, location)
    return pasted_imgs, img_masks

def past_SE(pasted_imgs, i, img_masks):
    random.seed(i+4)
    np.random.seed(i+4)
    num = random.randint(1, int(num_SE*1.5))
    num_SE_rand = np.random.choice(num_SE, num)+1
    for mask_SE in num_SE_rand:
        single_mask = Image.fromarray((labeled_SE==mask_SE)*1.0)
        single_bbox = single_mask.getbbox()
        SE_patch = SE.crop(single_bbox)
        SE_patchs, patch_masks, location = deformation(SE_patch, 15)
        pasted_imgs, img_masks = past(pasted_imgs, img_masks, SE_patchs, patch_masks, location)
    return pasted_imgs, img_masks

def run(i, image):
    random.seed(i)
    img_path1 = os.path.join(filepath1, image)
    img = Image.open(img_path1).convert("RGB")
    degree_level = random.randint(0, 100)
    # degree_level =2
    pasted_img = img.copy()
    pasted_imgs = [pasted_img]
    img_mask = Image.new("RGB", (img_size,img_size), (0, 0, 0))
    img_masks = [img_mask]
    pasted_imgs, img_masks =  past_MA(pasted_imgs, i, img_masks)#所有都先粘贴 MA
    if degree_level>degree_threshold[0]:
        print(image)
        pasted_imgs, img_masks =  past_HE(pasted_imgs, i, img_masks)
        pasted_imgs, img_masks =  past_EX(pasted_imgs, i, img_masks)
        pasted_imgs, img_masks =  past_SE(pasted_imgs, i, img_masks)
    elif degree_level>degree_threshold[1]:
        pasted_imgs, img_masks =  past_HE(pasted_imgs, i, img_masks)
        pasted_imgs, img_masks =  past_EX(pasted_imgs, i, img_masks)
    elif degree_level>degree_threshold[2]:
        pasted_imgs, img_masks =  past_HE(pasted_imgs, i, img_masks)
    paste_out_path =  os.path.join(out_path, image)
    pasted_imgs[0].save(paste_out_path, quality=100)

    mask_name = image.replace('jpeg', 'png')
    mask_out_path =  os.path.join(out_mask_path, mask_name)
    Lim_mask = img_masks[0].convert('L')#转灰度
    bim_mask = Lim_mask.point(table,'L')
    bim_mask.save(mask_out_path, quality=100)


if __name__ == '__main__':
    filepath1 = "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train/0/"#训练集中的正常的数据
    seg_path1 = "/mnt/huangwk/Dataset/idrid/seg_out/MA/"#各个类型病灶的分割结果
    seg_path2 = "/mnt/huangwk/Dataset/idrid/seg_out/HE/"
    seg_path3 = "/mnt/huangwk/Dataset/idrid/seg_out/EX/"
    seg_path4 = "/mnt/huangwk/Dataset/idrid/seg_out/SE/"
    out_path = "/mnt/huangwk/Dataset/idrid/paste_out_single_48_95_aug/"#粘贴图像输出路径
    out_mask_path = "/mnt/huangwk/Dataset/idrid/paste_out_single_48_95_aug_mask/"#粘贴mask输出路径
    name = 'IDRiD_48_'#48,39,13,49,03
    img_size=512
    blending_coef = 0.9
    degree_threshold = [95, 90, 80]
    source_path1 = os.path.join(seg_path1, name+'MA.tif')
    MA = Image.open(source_path1).convert("RGB")
    source_path2 = os.path.join(seg_path2, name+'HE.tif')
    HE = Image.open(source_path2).convert("RGB")
    source_path3 = os.path.join(seg_path3, name+'EX.tif')
    EX = Image.open(source_path3).convert("RGB")
    source_path4 = os.path.join(seg_path4, name+'SE.tif')
    SE = Image.open(source_path4).convert("RGB")
    threshold = 1
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(255)
    Lim_MA = MA.convert('L' )#转灰度
    bim_MA = Lim_MA.point(table,'L')#阈值分割
    labeled_MA, num_MA = label(np.asarray(bim_MA), background=0, return_num = True)#计算连通域，labeled_MA中不同的label代表不同的连通域
    Lim_HE = HE.convert('L' )
    bim_HE = Lim_HE.point(table,'L')
    labeled_HE, num_HE = label(np.asarray(bim_HE), background=0, return_num = True)
    Lim_EX = EX.convert('L' )
    bim_EX = Lim_EX.point(table,'L')
    labeled_EX, num_EX = label(np.asarray(bim_EX), background=0, return_num = True)
    Lim_SE = SE.convert('L' )
    bim_SE = Lim_SE.point(table,'L')
    labeled_SE, num_SE = label(np.asarray(bim_SE), background=0, return_num = True)
    augmenters = [iaa.Affine(scale={"x": (0.5, 1), "y": (0.5, 1)}),
                      iaa.Fliplr(0.5),
                      iaa.Flipud(0.5),
                      iaa.Affine(rotate=(-45, 45)),
                      iaa.GammaContrast((0.9,1.1),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.9,1.1),add=(-10,10)),#brightness
                      iaa.AddToHueAndSaturation((-10,10),per_channel=True)#color
                      ]
    
    img_names = os.listdir(filepath1)
    pool = Pool(processes=8)
    results = []
    for i, image in enumerate(img_names):
        results.append(pool.apply_async(run, (i, image)))

    for res in results:
        res.get()
