# uci_digits_gan.py

# GAN to generate synthetic '2' digits
# PyTorch 1.8.0-CPU Anaconda3-2020.02 (Python 3.7.6)
# Windows 10 

import numpy as np
import torch as T
import matplotlib as mpl
import matplotlib.pyplot as plt

from face_dataset import FastDataset
from network 

device = T.device("cpu") 

# -----------------------------------------------------------

class UCI_Digits_Dataset(T.utils.data.Dataset):
  # see Listing 1

class Generator(T.nn.Module):  # 20-40-64
  # see Listing 2

class Discriminator(T.nn.Module):  # 64-32-16-1
  # see Listing 3

# -----------------------------------------------------------

def accuracy(gen, dis, n, verbose=False): . . 

def display_digit(x, save=False): . . 

def main():
  # 0. get started
  print("Begin GAN for UCI 2 digits demo ")
  np.random.seed(0)
  T.manual_seed(0)
  np.set_printoptions(linewidth=36)
  mpl.rcParams['toolbar'] = 'None'

  # 1. create data objects
  print("Creating UCI Digits only-2s Dataset ")
  train_file = ".\\Data\\uci_digits_2_only.txt" 
  train_ds = UCI_Digits_Dataset(train_file)
  bat_size = 10
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True, drop_last=True)  

  # 1b. show typical training item (item [5])
  print("Typical training image (de-normed): ")
  digit = np.rint(train_ds[5].numpy() * 16)
  print(digit)
  display_digit(train_ds[5], save=False)

  # 2. create networks
  dis = Discriminator().to(device)  # 64-32-16-1
  gen = Generator().to(device)      # 20-40-64

  # 3. train GAN model
  max_epochs = 100
  ep_log_interval = 10
  lrn_rate = 0.005

  dis.train()  # set mode
  gen.train()
  dis_optimizer = T.optim.Adam(dis.parameters(), lrn_rate)
  gen_optimizer = T.optim.Adam(gen.parameters(), lrn_rate)
  loss_func = T.nn.BCELoss()
  all_ones = T.ones(bat_size, dtype=T.float32).to(device)
  all_zeros = T.zeros(bat_size, dtype=T.float32).to(device)

  print("Starting training ")
  for epoch in range(0, max_epochs):
    for (batch_idx, real_images) in enumerate(train_ldr):
      dis_accum_loss = 0.0  # to display progress
      gen_accum_loss = 0.0

      # 3a. train discriminator using real images
      dis_optimizer.zero_grad()
      dis_real_oupt = dis(real_images).reshape(-1)  # [0, 1]
      dis_real_loss = loss_func(dis_real_oupt,
        all_ones)  # or use squeeze()

      # 3b. train discriminator using fake images
      zz = T.normal(0.0, 1.0,
        size=(bat_size, gen.inpt_dim)).to(device)  # 10 x 20
      fake_images = gen(zz)
      dis_fake_oupt = dis(fake_images).reshape(-1)
      dis_fake_loss = loss_func(dis_fake_oupt, all_zeros)     
      dis_loss_tot = dis_real_loss + dis_fake_loss
      dis_accum_loss += dis_loss_tot

      dis_loss_tot.backward()  # compute gradients
      dis_optimizer.step()     # update weights and biases

      # 3c. train gen with fake images
      gen_optimizer.zero_grad()
      zz = T.normal(0.0, 1.0,
        size=(bat_size, gen.inpt_dim)).to(device)  # 20
      fake_images = gen(zz)
      dis_fake_oupt = dis(fake_images).reshape(-1)
      gen_loss = loss_func(dis_fake_oupt, all_ones)
      gen_accum_loss += gen_loss

      gen_loss.backward()
      gen_optimizer.step()

    if epoch % ep_log_interval == 0:
      acc_dis = Accuracy(gen, dis, 500, verbose=False)
      print(" epoch: %4d | dis loss: %0.4f | gen loss: %0.4f \
| dis accuracy: %0.4f "\
        % (epoch, dis_accum_loss, gen_accum_loss, acc_dis))

  print("Training complete ")

# -----------------------------------------------------------

  # 4. TODO: save trained model

  # 5. use generator to make fake images
  gen.eval()   # set mode
  for i in range(1):  # just 1 image for demo
    rinpt = T.randn(1, gen.inpt_dim).to(device)  # wrap normal()
    with T.no_grad():
      fi = gen(rinpt).numpy()  # make image, convert to numpy
    fi = np.rint(fi * 16)

    print("\nSynthetic generated image (de-normed): ")
    print(fi) 
    display_digit(fi)

# -----------------------------------------------------------

if __name__ == "__main__":
  main()