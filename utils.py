import numpy as np
import os
from fractions import Fraction

def calculate_rate(k, SNR, n=None):
    if k is None and n is None:
        print('must give either k or n')
        return
    # assum unit signal power
    P = 1.
    N0 = 10**(-(SNR/10))
    C = 0.5*np.log2(1+P/N0)

    if n is None:
        # determine the code length from the capacity
        n = int(np.ceil(k/C))
    elif k is None:
        k = int(np.floor(n*C))

    # the true rate is
    R = k/n
    
    return (k, n, R, C)
    
def calculate_mac_rate_pairs(n, snr1, snr2):
    k1, n, R1, C1 = calculate_rate(k=None, n=n, SNR=snr1)

    # assume unit signal power
    P1 = 1.
    N01 = 10**(-(snr1/10))

    C1 = 0.5*np.log2(1+P1/N01)
    k1 = int(np.floor(n*C1))
    R1 = k1/n
    
    # given the mac capacity, what is user 2s actual rate for a given k
    P2 = 1.
    N02 = 10**(-(snr2/10))
    C2 = 0.5*np.log2(1+(P2/N02)/(1+(P1/N01)))
    snr2_eff = 10*np.log10((P2/N02)/(1+(P1/N01)))
    # determine the alphabet size from the capacity
    k2 = int(np.floor(n*C2))

    # the true rate is
    R2 = k2/n
    
    return n, k1, R1, C1, k2, R2, C2

def calculate_reduced_rate(snr, max_n, l):
    k_orig, n_orig, r, c = calculate_rate(k=None, n=max_n, SNR=snr)
    k, n = Fraction(r).limit_denominator(max_n).as_integer_ratio()
    L = int(l/n)
    
    return k, n, L, r, c

def calculate_interference_ch(snr1, snr2):
    # calculate the noise power based on snr1
    N0 = 10**(-snr1/10)

    # calculate the scale factor for U2 st we have same N0
    h2 = N0*10**(snr2/10)

    # calculate the effective snr for U2
    snr2_eff = 10*np.log10(h2/(1+N0))
    
    return h2, snr2_eff
        
def plot_pentagon(n, snr1, snr2, fig=None, ax=None, do_r=True, do_tdma=True, do_text=True, points=['B', 'C', 'D']):
    import matplotlib.pyplot as plt
    _, _, r1, c1, _, r2_mac, c2_mac = calculate_mac_rate_pairs(n, snr1, snr2) # u1 at full rate
    _, _, r2, c2, _, r1_mac, c1_mac = calculate_mac_rate_pairs(n, snr2, snr1) # u2 at full rate
        
    # plot the region
    if fig is None or ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    ax.plot(c1*np.ones(50), np.linspace(0,c2_mac), 'k') #upright leg
    ax.plot(c1_mac*np.ones(50), np.linspace(0, c2), '--', color='gray') # ledge
    ax.plot(np.linspace(0, c1_mac), c2*np.ones(50), 'k') #roof
    ax.plot(np.linspace(0,c1), c2_mac*np.ones(50), '--', color='gray')
    ax.plot(np.linspace(c1_mac, c1), (c2_mac - c2)/(c1-c1_mac)*np.linspace(c1_mac, c1)+(c2+c1_mac), 'k') #dominant face
    if do_tdma:
        ax.plot(np.linspace(0,c1), (-c2/c1)*np.linspace(0,c1)+c2, ':', color='gray') # tdma
    
    if 'D' in points:
        ax.plot(c1, c2_mac, 'k*', label='Capacity')
        if do_text:
            ax.text(0.02+c1, c2_mac, 'D')
        if do_r:
            ax.plot(r1, r2_mac, 'p', color='gray', label='NN Code Rate $\\frac{k}{n}$' )
    if 'B' in points:
        ax.plot(c1_mac, c2, 'k*', label=None)
        if do_text:
            ax.text(0.02+c1_mac, c2, 'B')
        if do_r:
            ax.plot(r1_mac, r2, 'p', color='gray', label=None)
    if 'C' in points:
        c1_mid = c1_mac + 0.5*(c1-c1_mac)
        c2_mid = c2_mac + 0.5*(c2-c2_mac)
        ax.plot(c1_mid, c2_mid, 'k*', label=None)
        if do_text:
            ax.text(0.02+c1_mid, c2_mid, 'C')
    if 'E' in points:
        ax.plot(c1, 0, 'k*')
        if do_text:
            ax.text(0.02+c1, .02, 'E')
    if 'A' in points:
        ax.plot(0, c2, 'k*')
        if do_text:
            ax.text(0.02, 0.02+c2, 'A')
    
    return fig, ax

def create_model(args, load_weights=False):
    from models import SimpleAE
    model = SimpleAE(k=args.k, n=args.n, snr=args.snr, 
                    encoder_layers=args.encoder_layers, encoder_filters=args.encoder_filters, 
                    decoder_layers=args.decoder_layers, decoder_filters=args.decoder_filters, 
                    name=f'model_{args.experiment}_{args.test}')
    
    if load_weights:
        model.load_weights(os.path.join(args.test_dir, f'{args.experiment}_{args.test}_weights')).expect_partial()

    return model
