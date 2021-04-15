# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
import losses
from net import *
import numpy as np
from image_losses import get_loss

class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)

class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="",
                 encoder="", z_regression=False):
        super(Model, self).__init__()

        self.layer_count = layer_count
        self.z_regression = z_regression

        self.mapping_d = MAPPINGS["MappingD"](
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=3)

        self.mapping_f = MAPPINGS["MappingF"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.encoder = ENCODERS[encoder](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_f.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None, count=32, mixing=True, noise=True, return_styles=False, no_truncation=False):
        if z is None:
            z = torch.randn(count, self.latent_size)
        styles = self.mapping_f(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_f.num_layers, 1)

        if self.dlatent_avg_beta is not None: ## this is working on training -> and update self.dlatent_avg.buff.data register | but could not find where it will be using
            #print(f'1/ 3 ifs : self.dlatent_avg_beta is not None')
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)
        #else:print('NO : 1/ 3 ifs')

        if mixing and self.style_mixing_prob is not None:
            #print('2/ 3 ifs : mixing and self.style_mixing_prob is not None')
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                styles2 = self.mapping_f(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_f.num_layers, 1)

                layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)
        #else:print('NO : 2/ 3 ifs')
        if (self.truncation_psi is not None) and not no_truncation:
            #print('3/ 3 ifs : (self.truncation_psi is not None) and not no_truncation')
            layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)
        #else:print('NO : 3/ 3 ifs')
        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        if return_styles:
            return s, rec
        else:
            return rec

    def encode(self, x, lod, blend_factor):
        Z = self.encoder(x, lod, blend_factor)
        discriminator_prediction = self.mapping_d(Z)
        return Z[:, :1], discriminator_prediction

    def forward(self, x, lod, blend_factor, d_train, ae, loss_fracs=None, loss_types=None, noise_for_loss=None): 
        # blend_factor -> is not used for mixing styles but mixing different resolution levels in same image : search "ENCODERS/ DECODERS" in https://github.com/udithhaputhanthri/ALAE/blob/master/net.py
        if ae:
            #print('latent/ image space loss is computing started ... ')
            #print('injecting noise for image generation, reconstruction (inside "if ae :"): ',noise_for_loss[0], noise_for_loss[1])
            #print('dtype of noise variable [0], [1] (should be boolean) : ',type(noise_for_loss[0]), type(noise_for_loss[1]))
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            s, rec = self.generate(lod, blend_factor, z=z, mixing=False, noise=noise_for_loss[0], return_styles=True)

            Z, d_result_real = self.encode(rec, lod, blend_factor)

            assert Z.shape == s.shape

            if self.z_regression:
                raise NameError('Udith: z_regression=True and not implemented')
                Lae = torch.mean(((Z[:, 0] - z)**2))
                return Lae
            else:
                Lae = torch.mean(((Z - s.detach())**2))

                ## added by udith
                #print('*************************************************************************',Z.shape, s.shape)

                g_w = rec.detach() #self.decoder.forward(s.repeat(1, self.mapping_f.num_layers, 1), lod, blend_factor, True) ## CANT WE USE ABOVE-OBTAINED "REC" HERE 
                # above line : detach -> to break the cycle -> decoder, encoder part will be trained only once
                
                g_w.requires_grad=False # so g_w will not updated through image_space_loss
                Z_detached, _= self.encode(g_w, lod, blend_factor) 
                
                #print('*** G_W.requires_grad -> should be freezed (so should be: False) : ',g_w.requires_grad)
                #print('*** rec.requires_grad -> should not be freezed | because we are not using it (so should be: True) : ',rec.requires_grad)
                
                # below, the requires_grad= False for Generator network when generating g_w_hat:
                # Note that this and above-rec.detach() ensures that the weight updating due to image_space_loss (added new) only affects to the Encoder part (I -> w network) of the overall model
                # W want this because, when we do rec.detach(), we wanted to keep rec as constant. In order to do that, we should not update the generator (It will be updating through LAE and GAN losses)
                # To do that, we should remove the Generator from computational graph when decoding the w_hat

                for param in self.decoder.parameters(): # use model.decoder.parameters(): to access all parameters of model.decoder
                  param.requires_grad=False
                  #print('off gradient computation : ',param.requires_grad, param.shape)
                
                #print(f'X_detached.shape (should have 3 dims) : {Z_detached.shape}')
                g_w_hat  = self.decoder.forward(Z_detached.repeat(1, self.mapping_f.num_layers, 1), lod, blend_factor, noise_for_loss[1])
                
                for param in self.decoder.parameters(): # use model.decoder.parameters(): to access all parameters of model.decoder
                  param.requires_grad=True
                  #print('on gradient computation : ',param.requires_grad, param.shape)
                
                #print('*************************************************************************',g_w.shape, g_w_hat.shape)
                #raise NameError(f'Udith: Hi you are running well :: L1_image_loss : {l1_image} | total_loss : {Lae + l1_image}' )
                #print(f'Udith: you are running well :: L1_image_loss : {l1_image} | l1_W_loss (lae) : {Lae} | total_loss : {Lae + l1_image}')
                
                tot_loss=0

                summary=''
                for i in range(len(loss_types)):
                  loss_type = loss_types[i]
                  loss_frac = loss_fracs[i]
                  
                  if loss_type=='gan_g' or loss_type=='gan_d':continue
                  if loss_type =='lae':
                    summary += f" {loss_type} : {loss_frac*Lae} ::: "
                    tot_loss+= loss_frac*Lae
                  else:
                    summary += f' {loss_type} : {loss_frac*get_loss(g_w_hat, g_w , loss_type)} ::: '
                    tot_loss+= loss_frac*get_loss(g_w_hat, g_w , loss_type) # target is detached
                #print(summary)
                #print('total loss (without GAN loss) : ', tot_loss)  
                #print('latent/ image space loss is computing finished ... ')
                return tot_loss
                ## added by udith

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(lod, blend_factor, count=x.shape[0], noise=True)

            self.encoder.requires_grad_(True)

            _, d_result_real = self.encode(x, lod, blend_factor)

            _, d_result_fake = self.encode(Xp, lod, blend_factor)

            loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)
            
            #print('loss d : ', loss_fracs[loss_types.index('gan_d')]*loss_d)
            return loss_fracs[loss_types.index('gan_d')]*loss_d
        else:
            with torch.no_grad():
                z = torch.randn(x.shape[0], self.latent_size)

            self.encoder.requires_grad_(False)

            rec = self.generate(lod, blend_factor, count=x.shape[0], z=z.detach(), noise=True)

            _, d_result_fake = self.encode(rec, lod, blend_factor)

            loss_g = losses.generator_logistic_non_saturating(d_result_fake)
            
            #print('loss g : ',loss_fracs[loss_types.index('gan_g')]*loss_g)
            return loss_fracs[loss_types.index('gan_g')]*loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_d.parameters()) + list(self.mapping_f.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_d.parameters()) + list(other.mapping_f.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)

class GenModel(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="", encoder="", z_regression=False):
        super(GenModel, self).__init__()

        self.layer_count = layer_count

        self.mapping_f = MAPPINGS["MappingF"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_f.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None):
        styles = self.mapping_f(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_f.num_layers, 1)

        layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
        styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, True)
        return rec

    def forward(self, x):
        return self.generate(self.layer_count-1, 1.0, z=x)

