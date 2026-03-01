import os
from timm.models import load_checkpoint
import timm
from torch.utils.data import DataLoader
from multiscale_infer_dataset import InferDataset
from infer_methods import (
    gen_unc_vis,
    predict_varieties,
    gen_varieties_vis_and_hm,
    gen_probs_vis_and_hm,
)
import argparse
from openslide import OpenSlide
from classifier import MultiScaleClassifierFM
from postprocess import postprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="")
parser.add_argument("--varieties", type=str, default="norm,tumor")
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--crop_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--wsi_dirpath", type=str, required=True)
parser.add_argument("--mask_dirpath", type=str, required=True)
parser.add_argument("--thumb_dirpath", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--downsample", type=int, default=40)
parser.add_argument("--mask_resize", type=int, default=6)
parser.add_argument("--output_dirpath", type=str, default='output/')
parser.add_argument("--experiment", type=str, default="test")

args = parser.parse_args()

def gen_thumb():
    # Generate thumbnails
    for cur_dir, dirs, files in os.walk(args.wsi_dirpath):
        for file in files:
            if file.endswith(".tif"):
                wsi_path = os.path.join(cur_dir, file)
                slide = OpenSlide(wsi_path)
                slide_dimensions = slide.dimensions
                thumb_size = (int(slide_dimensions[0]/args.downsample), 
                              int(slide_dimensions[1]/args.downsample))
                thumb = slide.get_thumbnail(thumb_size)
                thumb_path = os.path.join(args.thumb_dirpath, file.replace(".tif", ".png"))
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                thumb.save(thumb_path)
    print("Thumbnail generation completed!")

def infer(
    model,
    varieties,
    wsi_path_list,
    mask_dirpath,
    thumb_dirpath,
    img_size,
    crop_size,
    downsample,
    mask_resize,
    infer_dirpath,
    gpu_id,
    batch_size,
    experiment,
    num_workers,
    save_npy=False,
):
    model.cuda(gpu_id)
    if save_npy:
        os.makedirs(os.path.join(infer_dirpath, "npy"), exist_ok=True)
    os.makedirs(os.path.join(infer_dirpath, "vis"), exist_ok=True)
    os.makedirs(os.path.join(infer_dirpath, "hm"), exist_ok=True)
    for i, wsi_path in enumerate(wsi_path_list):
        wsi_id = os.path.splitext(wsi_path.split("/")[-1])[0]
        for cur_dir, dirs, files in os.walk(mask_dirpath):
            for file in files:
                if file == wsi_id + ".png":
                    mask_path = os.path.join(cur_dir, file)
        for cur_dir, dirs, files in os.walk(thumb_dirpath):
            for file in files:
                if file == wsi_id + ".png":
                    thumb_path = os.path.join(cur_dir, file)
        npy_dirpath = os.path.join(infer_dirpath, "npy")
        vis_dirpath = os.path.join(infer_dirpath, "vis")
        hm_dirpath = os.path.join(infer_dirpath, "hm")

        # if vis path exists, pass it
        # if os.path.exists(os.path.join(vis_dirpath, "predict", wsi_id + ".png")):
        #     print(f"{wsi_id} exists, pass it")
        #     continue

        dataset = InferDataset(
            wsi_path, mask_path, img_size, crop_size, mask_resize, downsample
        )
        dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
        predict, probs, unc_map = predict_varieties(
            i + 1,
            len(wsi_path_list),
            model,
            len(varieties),
            dataloader,
            npy_dirpath,
            wsi_id,
            gpu_id,
            save_npy,
        )
        gen_varieties_vis_and_hm(
            predict, thumb_path, vis_dirpath, hm_dirpath, wsi_id, experiment
        )
        gen_probs_vis_and_hm(
            probs, varieties, thumb_path, vis_dirpath, hm_dirpath, wsi_id, experiment
        )
        gen_unc_vis(unc_map, thumb_path, vis_dirpath, npy_dirpath, wsi_id)


def main():
    # Load model
    print("loading model...")
    varieties = args.varieties
    varieties = varieties.split(",")
    ckpt_path = args.ckpt_path

    fm_kwargs = {
                'model_name': 'vit_large_patch16_224',
                'img_size': 224,
                'patch_size': 16,
                'init_values': 1e-5,
                'num_classes': 0,
                'dynamic_img_size': True
            }
    fm_model = timm.create_model(**fm_kwargs)

    model = MultiScaleClassifierFM(num_classes=2,
                 fm_model=fm_model,
                 hidden_size=1024,
                 num_heads=8,
                 dropout_rate=0.1)
    load_checkpoint(model, ckpt_path)
    model.eval()
    print("model load successfully!")


    gpu_id = args.gpu_id
    num_workers = args.num_workers
    current_wsi_path_list = []
    for cur_dir, dirs, files in os.walk(args.wsi_dirpath):
        for file in files:
            if file.endswith(".tif"):
                current_wsi_path_list.append(os.path.join(cur_dir, file))

    mask_dirpath = args.mask_dirpath
    thumb_dirpath = args.thumb_dirpath
    img_size = args.img_size
    crop_size = args.crop_size
    downsample = args.downsample
    mask_resize = args.mask_resize
    batch_size = args.batch_size
    output_dirpath = args.output_dirpath
    experiment = args.experiment

    os.makedirs(output_dirpath, exist_ok=True)

    t_start = time.time()
    infer(
        model,
        varieties,
        current_wsi_path_list,
        mask_dirpath,
        thumb_dirpath,
        img_size,
        crop_size,
        downsample,
        mask_resize,
        output_dirpath,
        gpu_id,
        batch_size,
        experiment,
        num_workers,
        save_npy=True
    )
    print("Total time:", time.time() - t_start)

if __name__ == "__main__":
    gen_thumb()
    main()
    postprocess(args.output_dirpath, [args.thumb_dirpath], args.experiment)
