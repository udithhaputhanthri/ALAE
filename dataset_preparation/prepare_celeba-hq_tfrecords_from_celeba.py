
import zipfile
import tqdm
from defaults import get_cfg_defaults
import sys
import logging
from net import *
import numpy as np
import random
import argparse
import os
import tensorflow as tf
import imageio
from PIL import Image
import cv2


def prepare_celeba(cfg, logger, train=True):
    if train:
        directory = os.path.dirname(cfg.DATASET.PATH)
    else:
        directory = os.path.dirname(cfg.DATASET.PATH_TEST)

    os.makedirs(directory, exist_ok=True)

    #archive = zipfile.ZipFile(os.path.join(directory, '/data/datasets/CelebA/Img/img_align_celeba.zip'), 'r')
    archive= zipfile.ZipFile('/home/udith/data/datasets/celeba-hq.zip', 'r')

    names = archive.namelist() #['content/celeba_small/000001.jpg', ...]
    names = [x for x in names if x[-4:] == '.jpg']  # ['content/celeba_small/000001.jpg', ...]

    if train:
        #names = [x for x in names if split_map[int(x[:-4][-6:])] != 2]
        names= names[:cfg.DATASET.SIZE]
    else:
        #names = [x for x in names if split_map[int(x[:-4][-6:])] == 2]
        names= names[cfg.DATASET.SIZE:]

    count = len(names)
    print("Count: %d" % count)

    #names = [x for x in names if x[-10:] not in corrupted]

    random.seed(0)
    random.shuffle(names)

    folds = cfg.DATASET.PART_COUNT
    celeba_folds = [[] for _ in range(folds)]

    ## removed if : UDITH
    count_per_fold = count // folds
    for i in range(folds):
        celeba_folds[i] += names[i * count_per_fold: (i + 1) * count_per_fold]
    ###

    for i in range(folds):
        images = []
        for x in tqdm.tqdm(celeba_folds[i]):  ## x-> is name eg: content/celeba_small/000001.jpg
            imgfile = archive.open(x)
            #image = center_crop(imageio.imread(imgfile.read()))
            reshape_size = 256 # 128 -> celeba, 256 -> celeba-hq
            image = cv2.resize(imageio.imread(imgfile.read()), (reshape_size, reshape_size))
            images.append((int(x[:-4][-6:]), image.transpose((2, 0, 1))))

        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

        if train:
            part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)
        else:
            part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)

        tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)

        random.shuffle(images)

        for label, image in images:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        tfr_writer.close()

        for j in range(6):
            images_down = []

            for label, image in tqdm.tqdm(images):
                h = image.shape[1]
                w = image.shape[2]
                image = torch.tensor(np.asarray(image, dtype=np.float32)).view(1, 3, h, w)

                image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8)

                image_down = image_down.view(3, h // 2, w // 2).numpy()
                images_down.append((label, image_down))

            if train:
                part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL - j - 1, i)
            else:
                part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL - j - 1, i)

            tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
            for label, image in images_down:
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
                tfr_writer.write(ex.SerializeToString())
            tfr_writer.close()

            images = images_down


def run():
    parser = argparse.ArgumentParser(description="ALAE. Prepare tfrecords for celeba128x128")
    parser.add_argument(
        "--config-file",
        default="configs/celeba-hq256.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    prepare_celeba(cfg, logger, True)
    prepare_celeba(cfg, logger, False)


if __name__ == '__main__':
    run()
