import torch.utils.data
from torchvision.utils import save_image
from net import *
import os
import utils
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader import *
from tqdm import tqdm
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
from model_with_newLoss import Model ## EDITED
from launcher import run
from defaults_newLoss import get_cfg_defaults
import lod_driver
from PIL import Image

from train_alae import save_sample

def train(cfg, logger, local_rank, world_size, distributed):
    loss_fracs= cfg.LOSS.fracs
    loss_types= cfg.LOSS.types

    torch.cuda.set_device(local_rank)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        z_regression=cfg.MODEL.Z_REGRESSION
    )
    model.cuda(local_rank)
    model.train()

    if local_rank == 0:
        model_s = Model(
            startf=cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=cfg.MODEL.LAYER_COUNT,
            maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=cfg.MODEL.MAPPING_LAYERS,
            channels=cfg.MODEL.CHANNELS,
            generator=cfg.MODEL.GENERATOR,
            encoder=cfg.MODEL.ENCODER,
            z_regression=cfg.MODEL.Z_REGRESSION)
        model_s.cuda(local_rank)
        model_s.eval()
        model_s.requires_grad_(False)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True)
        model.device_ids = None

        decoder = model.module.decoder
        encoder = model.module.encoder
        mapping_d = model.module.mapping_d
        mapping_f = model.module.mapping_f

        dlatent_avg = model.module.dlatent_avg
    else:
        decoder = model.decoder
        encoder = model.encoder
        mapping_d = model.mapping_d
        mapping_f = model.mapping_f
        dlatent_avg = model.dlatent_avg

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    decoder_optimizer = LREQAdam([
        {'params': decoder.parameters()},
        {'params': mapping_f.parameters()}
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': encoder.parameters()},
        {'params': mapping_d.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_d,
        'mapping_fl': mapping_f,
        'dlatent_avg': dlatent_avg
    }
    if local_rank == 0:
        model_dict['discriminator_s'] = model_s.encoder
        model_dict['generator_s'] = model_s.decoder
        model_dict['mapping_tl_s'] = model_s.mapping_d
        model_dict['mapping_fl_s'] = model_s.mapping_f

    tracker = LossTracker(cfg.OUTPUT_DIR)

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer,
                                    'scheduler': scheduler,
                                    'tracker': tracker
                                },
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = decoder.layer_to_resolution

    dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS)

    rnd = np.random.RandomState(3456)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    samplez = torch.tensor(latents).float().cuda()

    lod2batch = lod_driver.LODDriver(cfg, logger, world_size, dataset_size=len(dataset) * world_size)

    if cfg.DATASET.SAMPLES_PATH != 'no_path':
        path = cfg.DATASET.SAMPLES_PATH
        src = []
        with torch.no_grad():
            for filename in list(os.listdir(path))[:32]:
                img = np.asarray(Image.open(os.path.join(path, filename)))
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                im = img.transpose((2, 0, 1))
                x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
                if x.shape[0] == 4:
                    x = x[:3]
                src.append(x)
            sample = torch.stack(src)
    else:
        dataset.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, 32)
        sample = next(make_dataloader(cfg, logger, dataset, 32, local_rank))
        sample = (sample / 127.5 - 1.)

    lod2batch.set_epoch(scheduler.start_epoch(), [encoder_optimizer, decoder_optimizer])

    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [encoder_optimizer, decoder_optimizer])

        logger.info("Batch size: %d, Batch size per GPU: %d, LOD: %d - %dx%d, blend: %.3f, dataset size: %d" % (
                                                                lod2batch.get_batch_size(),
                                                                lod2batch.get_per_GPU_batch_size(),
                                                                lod2batch.lod,
                                                                2 ** lod2batch.get_lod_power2(),
                                                                2 ** lod2batch.get_lod_power2(),
                                                                lod2batch.get_blend_factor(),
                                                                len(dataset) * world_size))

        dataset.reset(lod2batch.get_lod_power2(), lod2batch.get_per_GPU_batch_size())
        batches = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), local_rank)

        scheduler.set_batch_size(lod2batch.get_batch_size(), lod2batch.lod)

        model.train()

        need_permute = False
        epoch_start_time = time.time()

        i = 0
        for x_orig in tqdm(batches):
            i += 1
            with torch.no_grad():
                if x_orig.shape[0] != lod2batch.get_per_GPU_batch_size():
                    continue
                if need_permute:
                    x_orig = x_orig.permute(0, 3, 1, 2)
                x_orig = (x_orig / 127.5 - 1.)

                blend_factor = lod2batch.get_blend_factor()

                needed_resolution = layer_to_resolution[lod2batch.lod]
                x = x_orig

                if lod2batch.in_transition:
                    needed_resolution_prev = layer_to_resolution[lod2batch.lod - 1]
                    x_prev = F.avg_pool2d(x_orig, 2, 2)
                    x_prev_2x = F.interpolate(x_prev, needed_resolution)
                    x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

            x.requires_grad = True

            encoder_optimizer.zero_grad()
            loss_d = model(x, lod2batch.lod, blend_factor, d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d))
            loss_d.backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            loss_g = model(x, lod2batch.lod, blend_factor, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g))
            loss_g.backward()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            lae = model(x, lod2batch.lod, blend_factor, d_train=True, ae=True, loss_fracs=loss_fracs, loss_types=loss_types)
            tracker.update(dict(lae=lae))
            lae.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if local_rank == 0:
                betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
                model_s.lerp(model, betta)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            lod_for_saving_model = lod2batch.lod
            lod2batch.step()
            if local_rank == 0:
                if lod2batch.is_time_to_save():
                    checkpointer.save("model_tmp_intermediate_lod%d" % lod_for_saving_model)
                if lod2batch.is_time_to_report():
                    save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s,
                                model.module if hasattr(model, "module") else model, cfg, encoder_optimizer,
                                decoder_optimizer)

        scheduler.step()

        if local_rank == 0:
            checkpointer.save("model_tmp_lod%d" % lod_for_saving_model)
            save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s,
                        model.module if hasattr(model, "module") else model, cfg, encoder_optimizer, decoder_optimizer)

    logger.info("Training finish!... save training results")
    if local_rank == 0:
        checkpointer.save("model_final").wait()

gpu_count = 1
run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/celeba-hq256-newLoss.yaml',
    world_size=gpu_count)
