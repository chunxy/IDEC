# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function, division
import argparse
import os
import numpy as np
from sklearn.cluster import BisectingKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear

from utils import Float32Dataset, MnistDataset, cluster_acc

# import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass



class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        # encoder
        enc = F.relu(self.enc_1(x))
        enc = F.relu(self.enc_2(enc))
        enc = F.relu(self.enc_3(enc))

        enc = self.z_layer(enc)

        # decoder
        dec = F.relu(self.dec_1(enc))
        dec = F.relu(self.dec_2(dec))
        dec = F.relu(self.dec_3(dec))
        dec = self.x_bar_layer(dec)

        return dec, enc


class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path='data/ae_mnist.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = AE(n_enc_1=n_enc_1,
                     n_enc_2=n_enc_2,
                     n_enc_3=n_enc_3,
                     n_dec_1=n_dec_1,
                     n_dec_2=n_dec_2,
                     n_dec_3=n_dec_3,
                     n_input=n_input,
                     n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', path)

    def forward(self, x):
        # print GPU memory
        print("IDEC forward", torch.cuda.memory_allocated())
        x_bar, z = self.ae(x)
        # cluster
        n_samples = z.size(0)
        n_clusters = self.cluster_layer.size(0)
        distances = torch.zeros(n_samples, n_clusters, device=z.device)
        print("IDEC forward computing distances", torch.cuda.memory_allocated())
        for i in range(n_clusters):
            distances[:, i] = torch.sum(torch.pow(z - self.cluster_layer[i], 2), dim=1)
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(200):
        total_loss = 0.
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
    torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train_idec():

    model = IDEC(n_enc_1=500,
                 n_enc_2=500,
                 n_enc_3=1000,
                 n_dec_1=1000,
                 n_dec_2=500,
                 n_dec_3=500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 alpha=1.0,
                 pretrain_path=args.pretrain_path).to(device)

    #  model.pretrain('data/ae_mnist.pkl')
    # if pretrain_path exists, load it
    if os.path.exists(args.pretrain_path):
        model.pretrain(args.pretrain_path)
    else:
        model.pretrain()

    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = dataset.data
    # y = dataset.y
    data = torch.Tensor(data).to(device)

    # Process data in batches to avoid GPU memory overflow
    batch_size = 1000
    n_samples = data.size(0)
    hidden_list = []
    x_bar_list = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = data[i:end]
            x_bar_batch, hidden_batch = model.ae(batch)
            hidden_list.append(hidden_batch.cpu())
            x_bar_list.append(x_bar_batch.cpu())

    hidden = torch.cat(hidden_list, dim=0)
    x_bar = torch.cat(x_bar_list, dim=0)

    print(f"Fitting kmeans with {args.n_clusters} clusters")
    kmeans = BisectingKMeans(n_clusters=args.n_clusters, init='k-means++', bisecting_strategy="biggest_inertia")
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    print("Kmeans fit done")
    # nmi_k = nmi_score(y_pred, y)
    # print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    for epoch in range(100):
        print(f"Epoch {epoch} of 100")
        if epoch % args.update_interval == 0:
            print("IDEC update", torch.cuda.memory_allocated())
            _, tmp_q = model(data)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            # acc = cluster_acc(y, y_pred)
            # nmi = nmi_score(y, y_pred)
            # ari = ari_score(y, y_pred)
            # print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
            #       ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        for batch_idx, (x, idx) in enumerate(train_loader):

            x = x.to(device)
            idx = idx.to(device)

            x_bar, q = model(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = args.gamma * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


datasets = {
  "gist": 960,
  "crawl": 300,
  "glove100": 100,
  "audio": 128,
  "video": 1024,
  "sift": 128,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--pretrain_path', type=str, default='data/ae_mnist')
    parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--n_samples', default=100000, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'mnist':
        args.pretrain_path = 'data/ae_mnist.pkl'
        args.n_clusters = 10
        args.n_input = 784
        dataset = MnistDataset()
    else:
        template_train = "/research/d1/gds/cxye23/datasets/data/{}_base.float32"
        template_model = "/research/d1/gds/cxye23/datasets/data/idec/ae_{}_{}.pkl"

        args.n_input = datasets[args.dataset]
        args.n_z = args.n_input // 2
        args.pretrain_path = template_model.format(args.dataset, args.n_clusters)
        dataset = Float32Dataset(template_train.format(args.dataset), args.n_input, args.n_samples)
    print(args)
    model = train_idec()

    with torch.no_grad():
        encoded_train = model.ae(dataset.full_data).cpu().numpy()
        encoded_train = encoded_train.reshape(-1, args.n_z).astype(np.float32)
        template_encoded_train = "/research/d1/gds/cxye23/datasets/data/idec/{}-{}.base.float32"
        encoded_train.tofile(template_encoded_train.format(args.dataset, args.n_clusters))

        template_query = "/research/d1/gds/cxye23/datasets/data/{}_query.float32"
        query = np.fromfile(template_query.format(args.dataset), dtype=np.float32)
        query = torch.from_numpy(query).to(device)
        encoded_query = model.ae(query).cpu().numpy()
        encoded_query = encoded_query.reshape(-1, args.n_z).astype(np.float32)
        template_encoded_query = "/research/d1/gds/cxye23/datasets/data/idec/{}-{}.query.float32"
        encoded_query.tofile(template_encoded_query.format(args.dataset, args.n_clusters))
