import torch.nn.functional as F
import numpy as np
import torch
import cv2
import os


def color_factory(type):
    if type == 0:
        return (0, 0, 0)
    elif type == 1:
        return (0, 0, 255) # blue
    elif type == 2:
        return (212, 54, 18)  # red
    elif type == 3:
        return (233, 240, 21)  # yellow
    else:
        raise Exception("color_factory break out!")


def predict_varieties(
    order,
    total,
    model,
    num_varieties,
    dataloader,
    npy_dirpath,
    wsi_id,
    gpu_id,
    save_npy,
):
    # Set background to 0, other classes start from 1
    predict = np.full_like(dataloader.dataset.mask, 0)
    probs = np.full(
        (
            dataloader.dataset.mask.shape[0],
            dataloader.dataset.mask.shape[1],
            num_varieties,
        ),
        0,
        dtype=np.float32,
    )
    unc_map = np.full_like(dataloader.dataset.mask, 0, dtype=np.float32)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img_10x = batch["img_10x"].cuda(gpu_id)
            img_4x = batch["img_4x"].cuda(gpu_id)
            mask_i = batch["mask_x"]
            mask_j = batch["mask_y"]
            
            data = (img_10x, img_4x)
            output, unc, _ = model(data)
            # predict: category map; probs: probability map
            predict_label = torch.argmax(output, dim=1)
            predict_label = predict_label.cpu().data.numpy().flatten()
            predict[mask_i, mask_j] = predict_label + 1
            probs_element = torch.softmax(output, dim=1)
            probs_element = probs_element.cpu().data.numpy()
            probs[mask_i, mask_j] = probs_element
            
            unc = torch.squeeze(unc)
            unc_map[mask_i, mask_j] = unc.cpu().data.numpy()
            print(
                "Total task: {}/{} Current task: {}/{}".format(order, total, i + 1, len(dataloader))
            )
    if save_npy:
        os.makedirs(os.path.join(npy_dirpath, "npy_predict"), exist_ok=True)
        os.makedirs(os.path.join(npy_dirpath, "npy_probs"), exist_ok=True)
        npy_predict_path = os.path.join(npy_dirpath, "npy_predict", wsi_id + ".npy")
        npy_probs_path = os.path.join(npy_dirpath, "npy_probs", wsi_id + ".npy")
        np.save(npy_predict_path, predict)
        np.save(npy_probs_path, probs)

    return predict, probs, unc_map


def gen_varieties_vis_and_hm(
    predict, thumbnail_path, vis_dirpath, hm_dirpath, wsi_id, experiment
):
    os.makedirs(os.path.join(vis_dirpath, "predict"), exist_ok=True)
    os.makedirs(os.path.join(vis_dirpath, "predict_pure"), exist_ok=True)
    os.makedirs(os.path.join(hm_dirpath, "predict"), exist_ok=True)
    vis_varieties_path = os.path.join(vis_dirpath, "predict", wsi_id + ".png")
    vis_varieties_pure_path = os.path.join(vis_dirpath, "predict_pure", wsi_id + ".png")
    hm_varieties_path = os.path.join(
        hm_dirpath, "predict", wsi_id + ".pred_" + experiment + ".png"
    )

    if type(predict) == str:
        predict = np.load(predict)

    thumbnail = cv2.imread(thumbnail_path)
    varieties_vis = np.zeros((predict.shape[0], predict.shape[1], 3), dtype="uint8")
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            varieties_vis[i, j] = color_factory(predict[i, j])

    varieties_vis = cv2.resize(
        varieties_vis,
        (thumbnail.shape[1], thumbnail.shape[0]),
    )

    varieties_vis = cv2.cvtColor(varieties_vis, cv2.COLOR_RGB2BGR)
    result = cv2.addWeighted(thumbnail, 0.7, varieties_vis, 0.5, 0)
    cv2.imwrite(vis_varieties_path, result)

    varieties_hm = cv2.cvtColor(varieties_vis, cv2.COLOR_BGR2BGRA)
    # Set the opacity of colored parts to 0.2, background to transparent
    varieties_hm[np.any(varieties_vis != (0, 0, 0), axis=2), 3] = 0.2 * 255
    varieties_hm[np.all(varieties_vis == (0, 0, 0), axis=2), 3] = 0

    cv2.imwrite(vis_varieties_pure_path, varieties_vis)
    cv2.imwrite(hm_varieties_path, varieties_hm)
    os.rename(hm_varieties_path, hm_varieties_path.replace(".png", ".hm"))


def gen_probs_vis_and_hm(
    probs, varieties, json_vis_path, vis_dirpath, hm_dirpath, wsi_id, experiment
):
    if type(probs) == str:
        probs = np.load(probs)
    probs = probs * 255
    for i, variety in enumerate(varieties):
        os.makedirs(os.path.join(vis_dirpath, "probs_" + variety), exist_ok=True)
        os.makedirs(os.path.join(hm_dirpath, "probs_" + variety), exist_ok=True)
        vis_path = os.path.join(vis_dirpath, "probs_" + variety, wsi_id + ".png")
        hm_path = os.path.join(
            hm_dirpath,
            "probs_" + variety,
            wsi_id + ".probs_" + variety + "_" + experiment + ".png",
        )

        own_probs = np.array(probs[:, :, i], dtype="uint8")
        own_probs = cv2.applyColorMap(own_probs, cv2.COLORMAP_JET)
        thumbnail = cv2.imread(json_vis_path)
        own_probs = cv2.resize(
            own_probs,
            (thumbnail.shape[1], thumbnail.shape[0])
        )
        result = cv2.addWeighted(thumbnail, 0.7, own_probs, 0.5, 0)
        cv2.imwrite(vis_path, result)

        probs_hm = cv2.cvtColor(own_probs, cv2.COLOR_BGR2BGRA)
        # Set the opacity of colored parts to 0.2, background to transparent
        probs_hm[np.any(own_probs != (0, 0, 0), axis=2), 3] = 0.2 * 255
        probs_hm[np.all(own_probs == (0, 0, 0), axis=2), 3] = 0
        cv2.imwrite(hm_path, probs_hm)
        os.rename(hm_path, hm_path.replace(".png", ".hm"))


def gen_unc_vis(unc_map, thumbnail_path, vis_dirpath, npy_dirpath, wsi_id):
    os.makedirs(os.path.join(npy_dirpath, "npy_unc"), exist_ok=True)
    unc_map_path = os.path.join(npy_dirpath, "npy_unc", wsi_id + ".npy")
    np.save(unc_map_path, unc_map)
    unc_map = unc_map * 255
    # Save unc overlay thumbnail
    os.makedirs(os.path.join(vis_dirpath, "unc"), exist_ok=True)
    # Save unc_pure
    os.makedirs(os.path.join(vis_dirpath, "unc_pure"), exist_ok=True)

    vis_unc_path = os.path.join(vis_dirpath, "unc", wsi_id + ".png")
    unc_pure_path = os.path.join(vis_dirpath, "unc_pure", wsi_id + ".png")
    unc_map = np.array(unc_map, dtype="uint8")

    unc_map = cv2.applyColorMap(unc_map, cv2.COLORMAP_JET)
    cv2.imwrite(unc_pure_path, unc_map)
    thumbnail = cv2.imread(thumbnail_path)
    unc_map = cv2.resize(
        unc_map,
        (thumbnail.shape[1], thumbnail.shape[0])
    )
    result = cv2.addWeighted(thumbnail, 0.7, unc_map, 0.5, 0)
    cv2.imwrite(vis_unc_path, result)



