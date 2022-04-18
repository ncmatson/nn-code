from itertools import product

from utils import calculate_rate, plot_pentagon
import numpy as np
import os
import tensorflow as tf
from models import AWGNLayer
import matplotlib.pyplot as plt

n = 60
L = 50
batch_size=500
snr_set = np.array([-1.5, 0., 1.5, 3.])
exp = 'mac_three_users_n48'
if os.environ['NN_CODE_ENV'] == 'm2':
    ONLINE=True
    data_dir = '/scratch/users/cmatson/nn-codes/data/'
elif os.environ['NN_CODE_ENV'] == 'local':
    ONLINE=False
    data_dir = '/Volumes/Elements/nn-codes/data/'

experiment_dir = os.path.join(data_dir, f'experiment_{exp}')

# create experiment params
def create_experiment():
    single_user_ex_set = []
    mac_set = []
    single_user_test = 0
    for snr1 in snr_set:
        k1, _, r1, c1 = calculate_rate(k=None, n=n, SNR=snr1)
        single_user_ex_set.append((single_user_test, k1, snr1, snr1, 1))
        u1_idx = single_user_test
        single_user_test+=1

        for snr2 in snr_set:
            N0 = 10**(-snr1/10)

            # calculate the scale factor and effective snr for U2 st we have same N0
            h2 = N0*10**(snr2/10)
            snr2_eff = 10*np.log10(h2/(1+N0))
            
            k2, _, r2, c2 = calculate_rate(k=None, n=n, SNR=snr2_eff)
            single_user_ex_set.append((single_user_test, k2, snr2, snr2_eff, 2))
            u2_idx = single_user_test
            single_user_test+=1
                

            for snr3 in snr_set:
                print(f'{snr1:.1f}, {snr2:.1f}, {snr3:.1f}')

                # calculate the scale factor and effective snr for U3 st we have same N0
                h3 = N0*10**(snr3/10)
                snr3_eff = 10*np.log10(h3/(1+h2+N0))
            
                k3, _, r3, c3 = calculate_rate(k=None, n=n, SNR=snr3_eff)
                single_user_ex_set.append((single_user_test, k3, snr3, snr3_eff, 3))
                u3_idx = single_user_test
                single_user_test +=1

                mac_set.append((u1_idx, u2_idx, u3_idx)) # append test ideces for individual tests

    return single_user_ex_set, mac_set
def load_model(idx):
    model = None
    if ONLINE:
        model_file = os.path.join(experiment_dir, f'test_{idx}', 'saved_model')

        model = tf.keras.models.load_model(model_file)
    return model

def perform_sic(m1, m2, m3, mac_params, iterations):
    if m3 == -1:
        return perform_sic2(m1, m2, mac_params, iterations)
    else:
        return perform_sic3(m1, m2, m3, mac_params, iterations)

def perform_sic2(m1, m2, mac_params, iterations):
    ber1 = np.random.random()
    ber2 = np.random.random()
    gp1 = np.random.random()
    gp2 = np.random.random()
    mse1 = np.random.random()
    mse2 = np.random.random()

    if ONLINE:
        acc2_fn = tf.keras.metrics.BinaryAccuracy()
        acc1_fn = tf.keras.metrics.BinaryAccuracy()
        mse1_fn = tf.keras.metrics.MeanSquaredError()
        mse2_fn = tf.keras.metrics.MeanSquaredError()

        n = mac_params['n']
        L = mac_params['L']
        k1 = mac_params['k1']
        k2 = mac_params['k2']
        snr1 = mac_params['snr1']
        h2 = mac_params['h2']

        rng = np.random.default_rng()
        
        @tf.function
        def sic2_step(b1, b2, iterations):
            x1 = m1.encoder(b1, training=False)
            x2 = m2.encoder(b2, training=False)

            x = x1 + np.sqrt(h2)*x2

            ch_mac = AWGNLayer(SNR=snr1)

            y = ch_mac(x)
            
            x1_hat = tf.zeros(shape=x1.shape)
            for i in tf.range(iterations):
                if i > 0:
                    # subtract for u2
                    y2_hat = y - x1_hat
                else:
                    # otherwise just start
                    y2_hat = y
                
                b2_hat = tf.expand_dims(m2.decoder(y2_hat, training=False), -1)

                # re-encode
                x2_hat = m2.encoder(b2_hat, training=False)

                # subtract for u1
                y1_hat = y - np.sqrt(h2)*x2_hat

                # decode user 1
                b1_hat = tf.expand_dims(m1.decoder(y1_hat, training=False), -1)
                
                # re-encode user 1
                x1_hat = m1.encoder(b1_hat, training=False)
                
                acc2_fn.update_state(b2, b2_hat)
                acc1_fn.update_state(b1, b1_hat)
                mse2_fn.update_state(x2, x2_hat)
                mse1_fn.update_state(x1, x1_hat)

        for _ in tf.range(50):
            b1 = rng.integers(0,2,size=(batch_size, k1*L, 1)).astype(np.float32)
            b2 = rng.integers(0,2,size=(batch_size, k2*L, 1)).astype(np.float32)
            
            sic2_step(b1, b2, iterations)

        ber1 = 1-float(acc1_fn.result())
        gp1 = k1/n * float(acc1_fn.result())
        ber2 = 1-float(acc2_fn.result())
        gp2 = k2/n * float(acc2_fn.result())
        mse2 = float(mse2_fn.result())
        mse1 = float(mse1_fn.result())
    
    return (ber1, ber2), (gp1, gp2), (mse1, mse2)


def perform_sic3(m1, m2, m3, mac_params, iterations):


    mac_ber = (np.random.random(), np.random.random(), np.random.random())
    mac_gp = (np.random.random(), np.random.random(), np.random.random())
    mac_mse = (np.random.random(), np.random.random(), np.random.random())
    rng = np.random.default_rng()
    
    if ONLINE:
        # metrics
        acc1_fn = tf.keras.metrics.BinaryAccuracy()
        acc2_fn = tf.keras.metrics.BinaryAccuracy()
        acc3_fn = tf.keras.metrics.BinaryAccuracy()
        mse1_fn = tf.keras.metrics.MeanSquaredError()
        mse2_fn = tf.keras.metrics.MeanSquaredError()
        mse3_fn = tf.keras.metrics.MeanSquaredError()

        k1 = mac_params['k1']
        k2 = mac_params['k2']
        k3 = mac_params['k3']
        n = mac_params['n']
        L = mac_params['L']
        snr1 = mac_params['snr1']
        h2 = mac_params['h2']
        h3 = mac_params['h3']
        
        # create the channel
        ch = AWGNLayer(SNR=snr1)
        
        @tf.function
        def sic3_step(b1, b2, b3, iterations):
            # encode
            x1 = m1.encoder(b1, training=False)
            x2 = m2.encoder(b2, training=False)
            x3 = m3.encoder(b3, training=False)

            # combine over the channel
            x = x1 + np.sqrt(h2)*x2 + np.sqrt(h3)*x3
            y = ch(x)


            # x1_hat, x2_hat, x3_hat = 0., 0., 0.
            x1_hat = tf.zeros(shape=x1.shape)
            x2_hat = tf.zeros(shape=x2.shape)
            x3_hat = tf.zeros(shape=x3.shape)

            for i in tf.range(iterations):
                y3_hat = y - x1_hat - np.sqrt(h2)*x2_hat # = y first time

                # decode u3, re-code, subtract
                b3_hat = m3.decoder(y3_hat, training=False)
                b3_hat = tf.expand_dims(b3_hat, -1)
                x3_hat = m3.encoder(b3_hat, training=False)
                y2_hat = y - x1_hat - np.sqrt(h3)*x3_hat

                # decode u2, re-code, subtract
                b2_hat = m2.decoder(y2_hat, training=False)
                b2_hat = tf.expand_dims(b2_hat, -1)
                x2_hat = m2.encoder(b2_hat, training=False)
                y1_hat = y - np.sqrt(h2)*x2_hat - np.sqrt(h3)*x3_hat

                # decode u1
                b1_hat = m1.decoder(y1_hat, training=False)
                b1_hat = tf.expand_dims(b1_hat, -1)
                x1_hat = m1.encoder(b1_hat, training=False)

                # update metrics
                acc1_fn.update_state(b1, b1_hat)
                acc2_fn.update_state(b2, b2_hat)
                acc3_fn.update_state(b3, b3_hat)

                mse1_fn.update_state(x1, x1_hat)
                mse2_fn.update_state(x2, x2_hat)
                mse3_fn.update_state(x3, x3_hat)
        
        # decode user3
        for step in range(50):
            b1 = rng.integers(0,2,size=(batch_size, k1*L, 1)).astype(np.float32)
            b2 = rng.integers(0,2,size=(batch_size, k2*L, 1)).astype(np.float32)
            b3 = rng.integers(0,2,size=(batch_size, k3*L, 1)).astype(np.float32)

            sic3_step(b1, b2, b3, iterations)
        
        # calculate final metrics
        ber1 = 1-float(acc1_fn.result())
        ber2 = 1-float(acc2_fn.result())
        ber3 = 1-float(acc3_fn.result())

        gp1 = k1/n*(1-ber1)
        gp2 = k2/n*(1-ber2)
        gp3 = k3/n*(1-ber3)

        mse1 = float(mse1_fn.result())
        mse2 = float(mse2_fn.result())
        mse3 = float(mse3_fn.result())

        mac_ber = (ber1, ber2, ber3)
        mac_gp = (gp1, gp2, gp3)
        mac_mse = (mse1, mse2, mse3)
    
    return mac_ber, mac_gp, mac_mse


        

###########################################################################################
# define experiments
if __name__ == '__main__':
    single_user_ex_set, mac_set = create_experiment()

    for test, (u1_idx, u2_idx, u3_idx) in enumerate(mac_set):
        if not ONLINE and test > 0:
            continue

        
        # construct the params
        k1, snr1 = single_user_ex_set[u1_idx][1:3] # test, k, snr, snr_eff, user
        k2, snr2, snr2_eff = single_user_ex_set[u2_idx][1:4]
        k3, snr3, snr3_eff = single_user_ex_set[u3_idx][1:4]
        print(f'3 user mac test #{test} ({u1_idx} {k1}, {u2_idx} {k2}, {u3_idx} {k3}):')
        print(f' SNRs = [{snr1:.3f}, {snr2:.3f}({snr2_eff:.3f}), {snr3:.3f}({snr3_eff:.3f})]')

        N0 = 10**(-snr1/10)
        h3 = N0*10**(snr3/10)
        h2 = N0*10**(snr2/10)

        sic_params={'k1':k1, 'k2':k2, 'k3':k3, 'snr1':snr1, 'h2':h2, 'h3':h3}
        _, _, r1, c1 = calculate_rate(k=None, n=n, SNR=snr1)
        _, _, r2, c2 = calculate_rate(k=None, n=n, SNR=snr2_eff)
        _, _, r3, c3 = calculate_rate(k=None, n=n, SNR=snr3_eff)
        mac_rates = (r1, r2, r3)
        mac_cap = (c1, c2, c3)

        # load models
        m1 = load_model(u1_idx)
        m2 = load_model(u2_idx)
        m3 = load_model(u3_idx)

        #####
        # perform SIC w/ 3 users, 1 iteration
        mac_ber3_it1, mac_gp3_it1, mac_mse3_it1 = perform_sic(m1, m2, m3, sic_params, iterations=1)
        print(f' 1 iteration SIC:')
        print(f'  ber: {mac_ber3_it1[0]:.4f}, {mac_ber3_it1[1]:.4f}, {mac_ber3_it1[2]:.4f}')
        print(f'  gp:  {mac_gp3_it1[0]:.4f}, {mac_gp3_it1[1]:.4f}, {mac_gp3_it1[2]:.4f}')
        print(f'  mse: {mac_mse3_it1[0]:.4f}, {mac_mse3_it1[1]:.4f}, {mac_mse3_it1[2]:.4f}')

        # 50 iterations
        mac_ber3_it50, mac_gp3_it50, mac_mse3_it50 = perform_sic(m1, m2, m3, sic_params, iterations=50)
        print(f' 50 iterations:')
        print(f'  ber: {mac_ber3_it50[0]:.4f}, {mac_ber3_it50[1]:.4f}, {mac_ber3_it50[2]:.4f}')
        print(f'  gp:  {mac_gp3_it50[0]:.4f}, {mac_gp3_it50[1]:.4f}, {mac_gp3_it50[2]:.4f}')
        print(f'  mse: {mac_mse3_it50[0]:.4f}, {mac_mse3_it50[1]:.4f}, {mac_mse3_it50[2]:.4f}')

        #####
        # perform SIC w/ only user 1 and 2, 1 iterations
        mac_ber2_it1, mac_gp2_it1, mac_mse2_it1 = perform_sic(m1, m2, -1, sic_params, iterations=1)
        print(f' two user results:')
        print(f'  1 iteration SIC:')
        print(f'   ber: {mac_ber2_it1[0]:.4f}, {mac_ber2_it1[1]:.4f}')
        print(f'   gp:  {mac_gp2_it1[0]:.4f}, {mac_gp2_it1[1]:.4f}')
        print(f'   mse: {mac_mse2_it1[0]:.4f}, {mac_mse2_it1[1]:.4f}')
        
        # 50 iterations
        mac_ber2_it50, mac_gp2_it50, mac_mse2_it50 = perform_sic(m1, m2, -1, sic_params, iterations=50)
        print(f' two user results:')
        print(f'  50 iterations:')
        print(f'   ber: {mac_ber2_it50[0]:.4f}, {mac_ber2_it50[1]:.4f}')
        print(f'   gp:  {mac_gp2_it50[0]:.4f}, {mac_gp2_it50[1]:.4f}')
        print(f'   mse: {mac_mse2_it50[0]:.4f}, {mac_mse2_it50[1]:.4f}')
        print('')
        
        
        ### plot
        ## 3 user results
        # plot 3 user results
        fig, ax = plt.subplots(tight_layout=True)
        ax.bar(np.arange(3)+1, mac_cap, label='mac capacity')
        ax.bar(np.arange(3)+1, mac_rates, label='mac rate k/n')
        ax.bar(np.arange(3)+1, mac_gp3_it50, label='goodput - 50 iter.')
        ax.bar(np.arange(3)+1, mac_gp3_it1, label='goodput - 1 iter.')
        ax.set_xticks(np.arange(3)+1)
        ax.set_xticklabels([f'{int(user)}\n{snr:.3f} ({snr_eff:.3f})' for snr, snr_eff, user in zip((snr1, snr2, snr3), (snr1, snr2_eff, snr3_eff), (1, 2, 3))])
        ax.set_xlabel('user, $SNR (SNR_{eff})$')
        ax.set_ylabel('rate')
        ax.legend()
        ax.set_title(f'Comparison of 3 User Rates\nTest #{test} $(n={n}, L={L})$')

        plt.savefig(os.path.join(experiment_dir, 'figures', f'three_user_rates_test_{test}_{n}_{L}_.png'), format='png')

        ## plot 2 user results on pentagon
        fig, ax = plot_pentagon(n, snr1, snr2)
        ax.plot(k1/n, k2/n, 'ko', label='network rate')
        ax.plot(mac_gp2_it1[0], mac_gp2_it1[1], 'bs', label='two user sic, 1 iter.')
        ax.plot(mac_gp3_it1[0], mac_gp3_it1[1], 'rs', label='three user sic, 1 iter.')
        ax.plot(mac_gp2_it50[0], mac_gp2_it50[1], 'bv', label='two user sic, 50 iter.')
        ax.plot(mac_gp3_it50[0], mac_gp3_it50[1], 'rv', label='three user sic, 50 iter.')
        ax.legend()
        ax.set_title(f'Comparison of Two vs. Three User SIC\nTest #{test} $(SNR_1={snr1}, SNR_2={snr2}, SNR_3={snr3}, n={n}, L={L})$')

        plt.savefig(os.path.join(experiment_dir, 'figures', f'two_vs_three_sic_test_{test}_{n}_{L}_.png'), format='png')

        if not ONLINE:
            plt.show()

        # print the differnece between the theoretical, max nn achievable, and actual sum rates
        sum_rate_cap = c1+c2+c3
        sum_rate_nn = r1+r2+r3
        sum_rate_gap_cap_nn = sum_rate_cap - sum_rate_nn
        
        sum_rate_gap_cap_it50 = sum_rate_cap - sum(mac_gp3_it50)
        sum_rate_gap_nn_it50 = sum_rate_nn - sum(mac_gp3_it50)
        
        sum_rate_gap_cap_it1 = sum_rate_cap - sum(mac_gp3_it1)
        sum_rate_gap_nn_it1 = sum_rate_nn - sum(mac_gp3_it1)
        
        print(f'capacity: {c1:.4f}, {c2:.4f}, {c3:.4f} - sum rate: {sum_rate_cap:.4f}') # capacity
        print(f'nn rates: {r1:.4f}, {r2:.4f}, {r3:.4f} - sum rate {sum_rate_nn:.4f}') # max achievable
        print(f'theor. gap: {c1-r1:.4f}, {c2-r2:.4f}, {c3-r3:.4f} - sum rate: {sum_rate_gap_cap_nn:.4f} ({sum_rate_gap_cap_nn/sum_rate_cap:.4f})') # individual gaps
        
        # print(f'gp 1: {mac_gp3_it1[0]:.4f}, {mac_gp3_it1[1]:.4f}, {mac_gp3_it1[2]:.4f} - sum rate: {sum(mac_gp3_it1):.4f}')
        print(f'gp 50: {mac_gp3_it50[0]:.4f}, {mac_gp3_it50[1]:.4f}, {mac_gp3_it50[2]:.4f} - sum rate: {sum(mac_gp3_it50):.4f}')
        print(f'gap to cap: {c1 - mac_gp3_it50[0]:.4f}, {c2 - mac_gp3_it50[1]:.4f}, {c3 - mac_gp3_it50[2]:.4f} - sum rate: {sum_rate_gap_cap_it50:.4f} ({sum_rate_gap_cap_it50/sum_rate_cap:.4f})')
