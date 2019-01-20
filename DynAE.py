import tensorflow as tf
import metrics
import math
import numpy as np
import matplotlib.pyplot as plt
import os

from time import time
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error
from datasets import generate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer, Flatten, Lambda, InputSpec, Input, Dense
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from datasets import generate_data_batch, generate_transformed_batch

PATH_RESULT = '/content/drive/My Drive/Colab/DynAE/results'
PATH_VIS = '/content/drive/My Drive/Colab/DynAE/visualisation'

def q_mat(X, centers, alpha=1.0):
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q**((alpha+1.0)/2.0)
        q = np.transpose(np.transpose(q)/np.sum(q, axis=1))
    return q

def generate_supervisory_signals(x_emb, x_img, centers_emb_fixed, centers_img_fixed, beta1, beta2):
    q = q_mat(x_emb, centers_emb_fixed, alpha=1.0)
    y_pred = q.argmax(1)
    confidence1 = q.max(1) 
    confidence2 = np.zeros((q.shape[0],))
    ind = np.argsort(q, axis=1)[:,-2]
    Y_encoder = []
    Y_autoencoder = []
    for i in range(x_img.shape[0]):
        confidence2[i] = q[i,ind[i]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
            Y_encoder.append(centers_emb_fixed[y_pred[i]])
            Y_autoencoder.append(centers_img_fixed[y_pred[i]])
        else:
            Y_encoder.append(x_emb[i])
            Y_autoencoder.append(x_img[i])    
    Y_autoencoder = np.asarray(Y_autoencoder)
    Y_encoder = np.asarray(Y_encoder)
    return Y_encoder, Y_autoencoder

def draw_centers(n_clusters, centers_img, img_h=28, img_w=28):
    plt.figure(figsize=(n_clusters, 4))
    for i in range(n_clusters):
        ax = plt.subplot(1, n_clusters, i + 1)
        plt.imshow(centers_img[i].reshape(img_h, img_w))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def total_loss(y_true, y_pred):
    return y_pred
 
def encoder_constructor(dims, visualisation_dir, act='relu'):
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    # input
    x = Input(shape=(dims[0],), name='input_encoder')
    h = x
    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)
    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here
    encoder = Model(inputs=x, outputs=h, name='encoder')
    plot_model(encoder, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcEncoder.png')
    return encoder

def decoder_constructor(dims, visualisation_dir, act='relu'):
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    # input
    z = Input(shape=(dims[-1],), name='input_decoder')
    y = z
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)
    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)
    decoder = Model(inputs=z, outputs=y, name='decoder')
    plot_model(decoder, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcDecoder.png')
    return decoder

def ae_constructor(encoder, decoder, dims, visualisation_dir):
    x = Input(shape=(dims[0],), name='input_autencoder')
    autoencoder = Model(inputs=x, outputs=decoder(encoder(x)), name='autoencoder')
    plot_model(autoencoder, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcAutoencoder.png')
    return autoencoder

def dynAE_constructor(encoder, ae, dims, visualisation_dir):
    input1 = Input(shape=(dims[0],), name='input_dynAE')
    target1 = Input(shape=(dims[-1],), name='target1_dynAE')
    target2 = Input(shape=(dims[0],), name='target2_dynAE')

    def loss_dynAE(x):
        encoder_output = x[0]
        ae_output = x[1]
        target1 = x[2]
        target2 = x[3]
        loss1 = tf.losses.mean_squared_error(encoder_output, target1)
        loss2 = tf.losses.mean_squared_error(ae_output, target2)
        return loss1 + loss2

    x1 = encoder(input1)
    x2 = ae(input1)
    out1 = Lambda(lambda x: loss_dynAE(x), name="output1_dynAE")((x1, x2, target1, target2))
    dynAE = Model(inputs=[input1, target1, target2], outputs=[out1], name='dynAE')
    plot_model(dynAE, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcDynAE.png')
    return dynAE

def i_ae_constructor(encoder, decoder, dims, visualisation_dir):
    input1 = Input(shape=(dims[0],), name='input_i_ae')
    alpha1 = Input(shape=(1,), name='alpha_i_ae')

    def mix_layer(x):
        encode = x[0]
        alpha1 = x[1]     
        encode_mix = alpha1 * encode + (1 - alpha1) * encode[::-1]   
        return encode_mix

    encode = encoder(input1)
    encode_mix = Lambda(mix_layer, name="mix_layer")((encode, alpha1))  
    decode_mix = decoder(encode_mix)
    i_ae = Model(inputs=[input1, alpha1], outputs=decode_mix, name='interpolation_autoencoder')
    plot_model(i_ae, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcI_AE.png')
    return i_ae
    
def critic_constructor(dims, visualisation_dir, act='relu'):
    n_stacks = len(dims) - 1 
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')    
    # input
    x = Input(shape=(dims[0],), name='input_critic')
    h = x
    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='critic_%d' % i)(h)
    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='critic_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here  
    h = Lambda(lambda x: tf.reduce_mean(x, [1]), name="reduce_mean_layer")(h)
    critic = Model(inputs=x, outputs=h, name='critic')
    plot_model(critic, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcCritic.png')
    return critic

def disc_constructor(critic, dims, visualisation_dir):
    input1 = Input(shape=(dims[0],), name='input1_disc')
    input2 = Input(shape=(dims[0],), name='input2_disc')
    alpha1 = Input(shape=(1,), name='alpha1_disc')
    alpha2 = Input(shape=(1,), name='alpha2_disc')
 
    def loss_disc(x):
        x1 = x[0]
        x2 = x[1]
        alpha1 = x[2]
        alpha2 = x[3]
        loss_within_cluster_interp = tf.reduce_mean(tf.square(x1 - alpha1))
        loss_between_cluster_interp = tf.reduce_mean(tf.square(x2 - alpha2))
        return loss_within_cluster_interp + loss_between_cluster_interp

    x1 = critic(input1)
    x2 = critic(input2)
    out1 = Lambda(loss_disc, name="output1_disc")((x1, x2, alpha1, alpha2))

    disc = Model(inputs=[input1, input2, alpha1, alpha2], outputs=[out1], name='discriminator')

    plot_model(disc, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcDiscriminator.png')
    return disc

def aci_ae_constructor(critic, ae, i_ae, advweight, dims, visualisation_dir):
    critic.trainable = False
    input1 = Input(shape=(dims[0],), name='input_aci_ae')
    alpha1 = Input(shape=(1,), name='alpha_aci_ae')

    def loss_aci_ae(x):
        input1 = x[0]
        x1 = x[1]
        x2 = x[2]
        loss_ae_critic = tf.reduce_mean(tf.square(x2))
        loss_ae = tf.losses.mean_squared_error(input1, x1)
        return loss_ae + advweight * loss_ae_critic

    x1 = ae(input1)
    x2 = critic(i_ae([input1, alpha1]))

    out1 = Lambda(lambda x: loss_aci_ae(x), name="output1_aci_ae")((input1, x1, x2))
    aci_ae = Model(inputs=[input1, alpha1], outputs=[out1], name='aci_ae')
    plot_model(aci_ae, show_shapes=True, show_layer_names=True, to_file=visualisation_dir + '/graph/FcACI_AE.png')
    return aci_ae

class DynAE:

    def __init__(self, batch_size, dataset, dims, loss_weight, n_clusters=10, alpha=1.0, visualisation_dir=PATH_VIS, ws=0.1, hs=0.1, rot=10, scale=0.0):
        self.batch_size = batch_size
        self.dataset = dataset
        self.dims = dims
        self.visualisation_dir = visualisation_dir
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.datagen = ImageDataGenerator(width_shift_range=ws, height_shift_range=hs, rotation_range=rot, zoom_range=scale)
        self.ws = ws 
        self.hs = hs
        self.rot = rot
        self.scale = scale
        #models
        self.encoder = encoder_constructor(self.dims, self.visualisation_dir)
        self.decoder = decoder_constructor(self.dims, self.visualisation_dir)
        self.ae = ae_constructor(self.encoder, self.decoder, self.dims, self.visualisation_dir)
        self.i_ae = i_ae_constructor(self.encoder, self.decoder, self.dims, self.visualisation_dir) 
        self.dynAE = dynAE_constructor(self.encoder, self.ae, self.dims, self.visualisation_dir)
        self.critic = critic_constructor(self.dims, self.visualisation_dir)
        self.disc = disc_constructor(self.critic, self.dims, self.visualisation_dir)
        self.aci_ae = aci_ae_constructor(self.critic, self.ae, self.i_ae, self.loss_weight, self.dims, self.visualisation_dir)
    
    def predict_ae(self, x):
        x_recons = self.ae.predict(x, verbose=0)
        return x_recons

    def predict_encoder(self, x):
        x_encode = self.encoder.predict(x, verbose=0)
        return x_encode

    def predict_i_ae(self, x, alpha):
        x_inter_recons = self.i_ae.predict([x, alpha], verbose=0)
        return x_inter_recons

    def compile_ae(self, optimizer='sgd'):
        self.ae.compile(optimizer=optimizer, loss='mse')

    def compile_dynAE(self, optimizer='sgd'):                                                                   
        self.dynAE.compile(optimizer=optimizer, loss=total_loss)

    def compile_critic(self, optimizer='sgd'):
        self.critic.compile(optimizer=optimizer, loss='mse')

    def compile_disc(self, optimizer='sgd'):
        self.disc.compile(optimizer=optimizer, loss=total_loss)

    def compile_aci_ae(self, optimizer='sgd'):
        self.aci_ae.compile(optimizer=optimizer, loss=total_loss)

    def train_on_batch_ae(self, x, y):
        return self.ae.train_on_batch(x, y)

    def train_on_batch_dynAE(self, x, y1, y2):
        y = np.zeros((x.shape[0],))
        return self.dynAE.train_on_batch([x, y1, y2], y)

    def train_on_batch_disc(self, x1, x2, y1, y2):
        y = np.zeros((x1.shape[0],))
        return self.disc.train_on_batch([x1, x2, y1, y2], y)

    def train_on_batch_aci_ae(self, x1, x2):
        y = np.zeros((x1.shape[0],))
        return self.aci_ae.train_on_batch([x1, x2], y)

    def train_aci_ae(self, x, y=None, maxiter=120e3, batch_size=256, validate_interval=2800, save_interval=2800, save_dir=PATH_RESULT, verbose=1, aug_train=True):
        print('Begin aci_ae training: ', '-' * 60)
        
        #Prepare log file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/' + self.dataset + '/train_aci_ae_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'loss_aci_ae', 'loss_disc'])
        logwriter.writeheader()
       
        #Initialization
        t0 = time()
        loss_aci_ae = 0
        loss_disc = 0
        index = 0
        index_array = np.arange(x.shape[0])
        
        #Training loop
        for ite in range(int(maxiter)):
            #Validation interval
            if ite % validate_interval == 0: 
                if y is not None and verbose > 0: 
                    avg_loss_aci_ae = loss_aci_ae / validate_interval 
                    avg_loss_disc = loss_disc / validate_interval   
                    loss_aci_ae = 0. 
                    loss_disc = 0.            
                    features = self.predict_encoder(x)
                    km = KMeans(n_clusters=self.n_clusters, n_init=20)
                    y_pred = km.fit_predict(features)
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    print('Iter %d: acc=%.5f, nmi=%.5f, loss_aci_ae=%.5f, loss_disc=%.5f' % (ite, acc, nmi, avg_loss_aci_ae, avg_loss_disc))
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, loss_aci_ae=avg_loss_aci_ae, loss_disc=avg_loss_disc)
                    logwriter.writerow(logdict)
                    logfile.flush()
            
            #Save interval
            if ite % save_interval == 0:
                self.ae.save_weights(save_dir + '/' + self.dataset + '/ae_weights.h5')
                self.critic.save_weights(save_dir + '/' + self.dataset + '/critic_weights.h5')

            #Train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            x_batch = x[idx]
            np.random.shuffle(x_batch)
            x_batch = self.random_transform(x_batch, ws=self.ws, hs=self.hs, rot=self.rot, scale=self.scale) if aug_train else x_batch
            alpha_interp = np.random.uniform(low=0.0, high=1.0, size=[x_batch.shape[0]])
            beta_interp = np.random.uniform(low=0.0, high=1.0, size=None)
            beta_interp = 0.5 - np.abs(beta_interp - 0.5)
            
            x_batch_recons = self.predict_ae(x_batch)
            x_batch_recons_interp = self.predict_i_ae(x_batch, alpha_interp)
            x_batch_mix = np.multiply(beta_interp, x_batch) + np.multiply(1 - beta_interp, x_batch_recons)

            loss1 = self.train_on_batch_aci_ae(x_batch, alpha_interp)
            loss2 = self.train_on_batch_disc(x_batch_recons_interp, x_batch_mix, alpha_interp, np.zeros((x_batch.shape[0],)))

            loss_aci_ae = loss_aci_ae + loss1
            loss_disc = loss_disc + loss2
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        logfile.close()
        print('training time: ', time() - t0)
        self.ae.save_weights(save_dir + '/' + self.dataset + '/ae_weights.h5')
        self.critic.save_weights(save_dir + '/' + self.dataset + '/critic_weights.h5')
        print('trained weights are saved to %s/%s/ae_weights.h5' % (save_dir, self.dataset))
        print('trained weights are saved to %s/%s/critic_weights.h5' % (save_dir, self.dataset))
        print('training: ', '-' * 60)

    def train_dynAE(self, x, y=None, kappa=3, n_clusters=10, maxiter=1e5, batch_size=256, tol=1e-2, validate_interval=140, show_interval=None, save_interval=2800, save_dir=PATH_RESULT, aug_train=True):
        #init
        number_of_samples = x.shape[0]
        img_h = int(math.sqrt(x.shape[1]))
        img_w = int(math.sqrt(x.shape[1]))

        #logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/' + self.dataset + '/train_dynAE_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'nb_unconf', 'nb_conf', 'loss'])
        logwriter.writeheader()

        #intervals config
        print('Begin clustering:', '-' * 60)
        if save_interval is None: 
            save_interval = int(maxiter)  # only save the initial and final model
        print('Save interval ', save_interval)
        if show_interval is None:
            show_interval = int(np.ceil(number_of_samples/batch_size))*20
        print('show interval ', show_interval)  

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        centers_emb, centers_img, y_pred, _ = self.generate_centers(x, n_clusters)
        y_pred_last = y_pred

        # Step 2: beta1 and beta2
        beta1, beta2 = self.generate_beta(kappa, n_clusters)

        # Step 3: deep clustering
        loss = 0
        index = 0
        nb_conf_prev = x.shape[0]
        index_array = np.arange(x.shape[0])
        delta_kappa = 0.3 * kappa
        for ite in range(int(maxiter)):
            if ite % validate_interval == 0:
                x_emb = self.encoder.predict(x)
                q = q_mat(x_emb, centers_emb)
                y_pred = q.argmax(1) 
                avg_loss = loss / validate_interval
                loss = 0.
                if ite > 0:
                    nb_conf_prev = nb_conf 
                nb_unconf, nb_conf = self.compute_nb_conflicted_data(x, centers_emb, beta1, beta2)
                #update centers
                if nb_conf >= nb_conf_prev:
                    centers_emb, centers_img, _, _ = self.generate_centers(x, n_clusters)
                    print("update principal centers")
                    beta1 = beta1 - (delta_kappa / n_clusters)
                    beta2 = beta2 - (delta_kappa / n_clusters)
                    delta_kappa = 0.3 * kappa
                    kappa = delta_kappa
                    print("update principal confidences")

                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)  
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, nb_unconf=nb_unconf, nb_conf=nb_conf, loss=avg_loss)
                    logwriter.writerow(logdict)
                    logfile.flush()
                    print('Iter %d: acc=%.5f, nmi=%.5f, ari=%.5f, nb_unconf=%d, nb_conf=%d,  loss=%.5f' % (ite, acc, nmi, ari, nb_unconf, nb_conf, avg_loss))
                    self.compute_acc_and_nmi_conflicted_data(x, y, centers_emb, beta1, beta2)
                    print("The number of unconflicted data points is : " + str(nb_unconf))
                    print("The number of conflicted data points is : " + str(nb_conf))
                y_pred_last = np.copy(y_pred)
                if(nb_conf / x.shape[0]) < tol:
                    logfile.close()
                    break

            if ite % show_interval == 0 and ite!=0:
                print("")
                print("----------------------------------------------------------------------------------")
                print("Centroids : ")
                print("----------------------------------------------------------------------------------")
                draw_centers(n_clusters, centers_img, img_h=img_h, img_w=img_w)
                
            # save intermediate model
            if ite % save_interval == 0:
                print("")
                print("----------------------------------------------------------------------------------")
                print("Save embeddings for visualization : ")
                print("----------------------------------------------------------------------------------")
                z = self.predict_encoder(x)
                q1 = q_mat(z, centers_emb)
                y1_pred = q1.argmax(1)

                pca = PCA(n_components=2).fit(z)
                z_2d = pca.transform(z)
                centers_2d = pca.transform(centers_emb)

                # save states for visualization
                np.save(self.visualisation_dir + '/embeddings/' + self.dataset + '/vis_' + str(ite) + '.npy', {'z_2d': z_2d, 'centers_2d': centers_2d, 'y_pred': y1_pred})

                print('saving model to: ', save_dir + '/' + self.dataset + '/ae_' + str(ite) + '.h5')
                self.ae.save_weights(save_dir + '/' + self.dataset + '/ae_weights_' + str(ite) + '.h5')

            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]

            X_img = x[idx]
            X_emb = self.predict_encoder(X_img)

            Y_encoder, Y_autoencoder = generate_supervisory_signals(X_emb, X_img, centers_emb, centers_img, beta1, beta2)
            X_transformed = self.random_transform(X_img, ws=self.ws, hs=self.hs, rot=self.rot, scale=self.scale) if aug_train else X_img

            losses = self.train_on_batch_dynAE(X_transformed, Y_encoder, Y_autoencoder)
            loss = loss + losses
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/' + self.dataset + '/ae_weights.h5')
        self.ae.save_weights(save_dir + '/' + self.dataset + '/ae_weights.h5')
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)

        return y_pred

    def generate_centers(self, x, n_clusters):
        features = self.predict_encoder(x)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10) 
        y_pred = kmeans.fit_predict(features)
        q = q_mat(features, kmeans.cluster_centers_, alpha=1.0)
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features)
        _, indices = nn.kneighbors(kmeans.cluster_centers_)
        centers_emb = np.reshape(features[indices], (-1, self.encoder.output.shape[1]))
        centers_img = self.decoder.predict(centers_emb)
        return centers_emb, centers_img, y_pred, q

    def generate_unconflicted_data_index(self, x_img, centers_emb, beta1, beta2):
        x_emb = self.encoder.predict(x_img)
        unconf_indices = []
        conf_indices = []
        q = q_mat(x_emb, centers_emb, alpha=1.0)
        confidence1 = q.max(1)
        confidence2 = np.zeros((q.shape[0],))
        a = np.argsort(q, axis=1)[:,-2]  
        for i in range(x_img.shape[0]):
            confidence2[i] = q[i,a[i]]
            if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
                unconf_indices.append(i)
            else:
                conf_indices.append(i)
        unconf_indices = np.asarray(unconf_indices, dtype=int)
        conf_indices = np.asarray(conf_indices, dtype=int)
        return unconf_indices, conf_indices

    def compute_acc_and_nmi_conflicted_data(self, x, y, centers_emb, beta1, beta2):
        features = self.predict_encoder(x)
        unconf_indices, conf_indices = self.generate_unconflicted_data_index(x, centers_emb, beta1, beta2)
        
        if unconf_indices.size == 0:
            print(' '*8 + "Empty list of unconflicted data")
        else:
            x_emb_unconf = self.predict_encoder(x[unconf_indices])
            y_unconf = y[unconf_indices]
            y_pred_unconf = q_mat(x_emb_unconf, centers_emb, alpha=1.0).argmax(axis=1)
            print(' '*8 + '|==>  acc unconflicted data: %.4f,  nmi unconflicted data: %.4f  <==|'% (metrics.acc(y_unconf, y_pred_unconf), metrics.nmi(y_unconf, y_pred_unconf)))

        if conf_indices.size == 0:
            print(' '*8 + "Empty list of conflicted data")
        else:
            x_emb_conf = self.predict_encoder(x[conf_indices])
            y_conf = y[conf_indices]
            y_pred_conf = q_mat(x_emb_conf, centers_emb, alpha=1.0).argmax(axis=1)
            print(' '*8 + '|==>  acc conflicted data: %.4f,  nmi conflicted data: %.4f  <==|'% (metrics.acc(y_conf, y_pred_conf), metrics.nmi(y_conf, y_pred_conf)))
    
    def compute_acc_and_nmi(self, x, y):
        features = self.predict_encoder(x)
        km = KMeans(n_clusters=len(np.unique(y)), n_init=20)
        y_pred = km.fit_predict(features)
        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'% (metrics.acc(y, y_pred), metrics.nmi(y, y_pred)))

    def compute_nb_conflicted_data(self, x, centers_emb, beta1, beta2):
        unconf_indices, conf_indices = self.generate_unconflicted_data_index(x, centers_emb, beta1, beta2)
        return unconf_indices.shape[0], conf_indices.shape[0]
        
    def random_transform(self, x, ws=0.1, hs=0.1, rot=10, scale=0.0):
        self.datagen = ImageDataGenerator(width_shift_range=ws, height_shift_range=hs, rotation_range=rot, zoom_range=scale)
        if len(x.shape) > 2:  # image
            return self.datagen.flow(x, shuffle=False, batch_size=x.shape[0]).next()

        # if input a flattened vector, reshape to image before transform
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen = self.datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=x.shape[0])
        return np.reshape(gen.next(), x.shape)     

    def generate_beta(self, kappa, n_clusters):
        beta1 = kappa / n_clusters
        beta2 = beta1 / 2 
        print("Beta1 = " + str(beta1) + " and Beta2 = " + str(beta2))
        return beta1, beta2
