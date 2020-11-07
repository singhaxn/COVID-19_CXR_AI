import sys, os, gc
from functools import partial
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import argparse
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fastai.vision.all import get_image_files, load_learner, Image, F
from segmentation_data import find_bounds, compute_confidence_score
from classifier_common import *
from heatmaps import *

def save_heatmap_sbs(hmconfig, xray_file, cam, xclass, conf, cropper, heatmap_file):
    img = Image.open(xray_file)
    
    figsize = hmconfig["figsize"]
    fig, ax = plt.subplots(1, 2, figsize=(figsize*2, figsize))
    fig.patch.set_facecolor('white')
    
    ax[0].axis("off")
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("X-Ray")
    
    x, y, width, height = cropper.get_crop_bounds(xray_file, img.size)
    
    heatmap = cv2.resize(
        cam.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )
    
    ax[1].axis("off")
    ax[1].set_title(f"{xclass} ({conf:.2})")
    ax[1].imshow(heatmap, extent=(x, x+width, y+height, y), cmap=hmconfig["cmap"])
    ax[1].imshow(img, cmap="gray", alpha=hmconfig["xray-opacity"])
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='yellow', facecolor='none')
    ax[1].add_patch(rect)
    
    fig.savefig(heatmap_file, bbox_inches='tight')
    plt.close(fig)

def save_heatmap(hmconfig, xray_file, cam, xclass, conf, cropper, heatmap_file):
    img = Image.open(xray_file)
    
    figsize = hmconfig["figsize"]
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    fig.patch.set_facecolor('white')
    
    x, y, width, height = cropper.get_crop_bounds(xray_file, img.size)
    
    heatmap = cv2.resize(
        cam.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )
    
    ax.axis("off")
    ax.set_title(f"{xclass} ({conf:.2})")
    ax.imshow(heatmap, extent=(x, x+width, y+height, y), cmap=hmconfig["cmap"])
    ax.imshow(img, cmap="gray", alpha=hmconfig["xray-opacity"])
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)
    
    fig.savefig(heatmap_file, bbox_inches='tight')
    plt.close(fig)

def save_heatmaps(hmconfig, xray_files, cams, classes, confs, cropper, heatmap_dir):
    for xray_file, cam, xclass, conf in zip(xray_files, cams, classes, confs):
        heatmap_file = os.path.join(heatmap_dir, f"{xray_file.stem}.png")
        save_heatmap(hmconfig, xray_file, cam, xclass, conf, cropper, heatmap_file)

class HeatmapMPWrapper:
    def __init__(self, data):
        self.data = data
    
    def save_heatmaps_thread(self, xray_files):
        for xray_file in xray_files:
            save_heatmap(*self.data[xray_file])

def save_heatmaps_mp(hmconfig, xray_files, cams, classes, confs, cropper, heatmap_dir):
    n_files = len(xray_files)
    data = {}
    for xray_file, cam, xclass, conf in zip(xray_files, cams, classes, confs):
        heatmap_file = os.path.join(heatmap_dir, f"{xray_file.stem}.png")
        data[xray_file] = (hmconfig, xray_file, cam, xclass, conf, cropper, heatmap_file)
    
    n_cores = mp.cpu_count()
    chunk_size = n_files // n_cores
    chunk_excess = n_files % n_cores
    split = []
    start = 0
    for i in range(n_cores):
        cs = (chunk_size + 1) if i < chunk_excess else chunk_size
        split.append(xray_files[start:start+cs])
        start += cs
    
    # Save and restore the clahe object because
    # it can't be pickled for parallel processing
    saved_clahe = cropper.clahe
    cropper.clahe = None
    
    pool = mp.Pool(n_cores)
    hmpw = HeatmapMPWrapper(data)
    pool.map(hmpw.save_heatmaps_thread, split)
    pool.close()
    pool.join()
    
    cropper.clahe = saved_clahe

def segment(config, img_files, show_progress=False):
    segmenter = load_learner(config["segmenter"])
    segmenter.freeze()
    
    step_size = config["batch_size"]
    n_files = len(img_files)
    regions = []
    
    for i in range(0, n_files, step_size):
        i_end = min(i+step_size, n_files)
        xray_files = img_files[i:i_end]
        dl = segmenter.dls.test_dl(xray_files, bs=step_size)
        inputs, preds, _, masks = segmenter.get_preds(dl=dl, with_input=True, with_decoded=True)
        
        for in_file, img, pred, mask in zip(xray_files, inputs, preds, masks):
            bounds = find_bounds(mask, padding=0.05)
            confidence = compute_confidence_score(pred, mask)
            im = Image.open(in_file)
            width, height = im.size

            real_bounds = (
                int(bounds[0] / mask.shape[1] * width),
                int(bounds[1] / mask.shape[0] * height),
                int((bounds[2] - bounds[0]) / mask.shape[1] * width),
                int((bounds[3] - bounds[1]) / mask.shape[0] * height)
            )
            regions.append((str(in_file), *real_bounds, confidence))
        
        if show_progress:
            print(f"{i_end} / {n_files}")
    
    del segmenter
    gc.collect()
    
    regions_df = pd.DataFrame(columns=[
        "file", "x", "y", "width", "height", "confidence"
    ], data=regions).set_index("file")
    
    return regions_df

def _get_cropped_image(cropper, im_file):
    return cropper.get_cropped_image(str(im_file))

get_cropped_image = None

def classify(config, img_files, regions_df, heatmap_dir=None, show_progress=False):
    global get_cropped_image
    
    cropper = InputCropper(regions_df, config["crop_confidence_threshold"])
    get_cropped_image = partial(_get_cropped_image, cropper)
    
    classifier = load_learner(config["classifier"])
    classifier.freeze()    
    vocab = classifier.dls[0].vocab
    
    step_size = config["batch_size"]
    hmconfig = config["heatmap"]
    n_files = len(img_files)
    predictions = []
    confidences = []
    seg_confidences = []
    
    with HeatmapGenerator(classifier.model[0]) as hmgen:
        for i in range(0, n_files, step_size):
            i_end = min(i+step_size, n_files)

            xray_files = img_files[i:i_end]
            dl = classifier.dls.test_dl(xray_files, bs=step_size)

            for x, in dl:
                output = classifier.model.eval()(x)
                preds = F.softmax(output, dim=1)
                classes = preds.argmax(dim=1)
                conf = preds[np.arange(preds.shape[0]), classes].tolist()
                readable_classes = [vocab[c] for c in classes]
                predictions.extend(readable_classes)
                confidences.extend(conf)

                if heatmap_dir:
                    output.backward(output)
                    cam_map = hmgen.get_heatmaps()
                    save_heatmaps_mp(hmconfig, xray_files, cam_map, readable_classes, conf,
                                  cropper, heatmap_dir)
            
            seg_confidences.extend(
                [regions_df.loc[str(f)]["confidence"] for f in xray_files]
            )

            if show_progress:
                sys.stdout.write(f"\r{i_end} / {n_files}")
    
    del classifier
    gc.collect()
    
    return predictions, confidences, seg_confidences

'''
Run inference on all images (JPEG / PNG) in a directory (non-recursive)
Parameters
    config_path: The location of the config JSON file for the model to be used for inference
    xray_path: Directory containing a set of (uploaded) X-Ray images (JPEG / PNG)
    heatmap_path: [optional] Directory to store the generated heatmaps. File names will be the same as those of the input files. By default, heatmaps are not generated
Return
    An iterable that produces tuples of the form (filename, prediction, confidence, seg_confidence)
    filename: name of the image file
    prediction: one of ['normal', 'pneumonia', 'COVID-19']
    confidence: float in the range [0.0, 1.0] indicating the confidence level for this prediction
    seg_confidence: confidence of the segmentation stage. May indicate image quality.
'''
def infer_dir(config_path, xray_path, heatmap_path=None, show_progress=False):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    img_files = sorted(get_image_files(xray_path, recurse=False))
    
    # Locate lungs
    regions_df = segment(config, img_files, show_progress)
    
    # Classify
    predictions, confidences, seg_confidences = classify(
        config, img_files, regions_df, heatmap_path, show_progress
    )
    
    img_filenames = [f"{f.stem}{f.suffix}" for f in img_files]
    
    return zip(img_filenames, predictions, confidences, seg_confidences)

'''
Run inference on a single image (JPEG / PNG)
Parameters
    config_path: The location of the config JSON file for the model to be used for inference
    xray_path: Path to an (uploaded) X-Ray image (JPEG / PNG)
    heatmap_path: [optional] Directory to store the generated heatmaps. File names will be the same as those of the input files. By default, heatmaps are not generated
Return
    A tuple of the form (prediction, confidence, seg_confidence)
    prediction: one of ['normal', 'pneumonia', 'COVID-19']
    confidence: float in the range [0.0, 1.0] indicating the confidence level for this prediction
    seg_confidence: confidence of the segmentation stage. May indicate image quality.
'''
def infer(config_path, xray_path, heatmap_path=None):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    img_file = Path(xray_path)
    
    # Locate lungs
    regions_df = segment(config, [img_file], False)
    
    # Classify
    predictions, confidences, seg_confidences = classify(
        config, [img_file], regions_df, heatmap_path, False
    )
    
    return predictions[0], confidences[0], seg_confidences[0]


if __name__ == '__main__':
    # Example: python3 inference.py --config=model-config.json --imagepath="/data/datasets/extended_v2_Masked/test/COVID-19/CR.1.2.840.113564.1722810170.20200330111806125940.1003000225002.png"
    # Example: python3 inference.py --config=model-config.json --imagepath="/data/datasets/extended_v2_Masked/test/Normal/0f8c91da-7e03-480e-8760-1604b1d53c97.png"
    # Example: python3 inference.py --config=model-config.json --imagepath="/data/datasets/extended_v2/evaluate --heatmappath=/data/heatmaps/evaluate"
    
    parser = argparse.ArgumentParser(description='COVID-19_CXR_AI Inference')
    
    parser.add_argument('--config', default=None, required=True, type=str, help='Config file path')
    parser.add_argument('--xraypath', default=None, required=True, type=str, help='Full path to image (or dir containing images) to be inferenced')
    parser.add_argument('--heatmappath', default=None, type=str, help='Directory in which generated heatmaps are to be stored')
    args = parser.parse_args()
    
    if args.heatmappath:
        os.makedirs(args.heatmappath, exist_ok=True)
    if os.path.isdir(args.xraypath):
        results = infer_dir(args.config, args.xraypath, args.heatmappath, True)
        n_results = 0

        if args.heatmappath:
            results_file = os.path.join(args.heatmappath, "results.csv")
        else:
            results_file = "/tmp/results.csv"
        
        with open(results_file, "w") as results_csv:
            print("Image,Prediction,Confidence,Estimated Quality", file=results_csv)
            for result in sorted(results, key=lambda x: x[0]):
                print(result)
                print(",".join([str(v) for v in result]), file=results_csv)
                n_results += 1
    else:
        p, c, q = infer(args.config, args.xraypath, args.heatmappath)
        print(args.xraypath, p, c, q)