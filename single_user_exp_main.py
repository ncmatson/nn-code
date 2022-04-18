from argparse import ArgumentParser
import os

import numpy as np
import tensorflow as tf

from utils import calculate_rate
from models import SimpleAE
import json

def is_progress(val_history, start_considering):
    loss = np.array(val_history['loss'])
    min_loss_idx = np.argmin(loss[start_considering:])
    # if its been at least 10 epochs since the min loss
    if ((len(loss)-1) - (min_loss_idx + start_considering)) > 10:
        return False
    else:
        return True


def run_experiment(args):
    ###
    # parse some args
    ###
    
    calc_k, _, calc_r, _ = calculate_rate(k=None, n=args.n, SNR=args.snr)
    if args.k is None:
        args.k = calc_k
        args.rate = calc_r
    else:
        args.rate = args.k/args.n

    ###
    # build the encoder and decoder
    ###
    print('building model...')

    model = SimpleAE(k=args.k, n=args.n, snr=args.snr, 
                    encoder_layers=args.encoder_layers, encoder_filters=args.encoder_filters, 
                    decoder_layers=args.decoder_layers, decoder_filters=args.decoder_filters, 
                    name=f'model_{args.experiment}_{args.test}')

    optimizer_e = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    acc_fn_e = tf.keras.metrics.BinaryAccuracy()
    acc_fn_d = tf.keras.metrics.BinaryAccuracy()
    acc_fn_v = tf.keras.metrics.BinaryAccuracy()

    ###
    # train the network
    ###
    
    # train
    print('training...')
    
    rng = np.random.default_rng()
    updates = 0
    last_check_point = 0
    
    encoder_history = {'loss':[], 'binary_accuracy':[]}
    decoder_history = {'loss':[], 'binary_accuracy':[]}
    val_history = {'loss':[], 'binary_accuracy':[]}
    
    @tf.function
    def train_encoder(b):
        with tf.GradientTape() as tape:
            x = model.encoder(b, training=True)
            y = model.ch(x)
            b_hat = model.decoder(y, training=False)

            loss = loss_fn(b, b_hat)
            acc_fn_e.update_state(b, b_hat)
        
        grads = tape.gradient(loss, model.encoder.trainable_weights)
        optimizer_e.apply_gradients(zip(grads, model.encoder.trainable_weights))

        return loss, float(acc_fn_e.result())

    @tf.function
    def train_decoder(b):
        with tf.GradientTape() as tape:
            x = model.encoder(b, training=False)
            y = model.ch(x)
            b_hat = model.decoder(y, training=True)

            loss = loss_fn(b, b_hat)
            acc_fn_d.update_state(b, b_hat)
        
        grads = tape.gradient(loss, model.decoder.trainable_weights)
        optimizer_d.apply_gradients(zip(grads, model.decoder.trainable_weights))

        return loss, float(acc_fn_d.result())

    @tf.function
    def val_step(b):
        x = model.encoder(b, training=False)
        y = model.ch(x)
        b_hat = model.decoder(y, training=False)

        loss = loss_fn(b, b_hat)
        acc_fn_v.update_state(b, b_hat)
        

        return loss, float(acc_fn_v.result())

    for epoch in range(args.epochs):
        print(f'epoch {epoch+1}:')
        # train the encoder
        for step in range(args.encoder_steps):
            b = rng.integers(0,2,size=(args.batch_size, args.k*args.block_len, 1)).astype(np.float32)
            loss_e, acc_e = train_encoder(b)
#            print(f' step {step+1}: {loss_e:.3f}', end='\r')
        encoder_history['loss'].append(loss_e)
        encoder_history['binary_accuracy'].append(acc_e)
        print(f' encoder: loss {loss_e:.4f}, acc {acc_e:.4f}')

        # train the decoder
        for step in range(args.decoder_steps):
            b = rng.integers(0,2,size=(args.batch_size, args.k*args.block_len, 1)).astype(np.float32)
            loss_d, acc_d = train_decoder(b)
 #           print(f' step {step+1}: {loss_d:.3f}', end='\r')

        decoder_history['loss'].append(loss_d)
        decoder_history['binary_accuracy'].append(acc_d)
        print(f' decoder: loss {loss_d:.4f}, acc {acc_d:.4f}')

        # validate
        for step in range(args.val_steps):
            b = rng.integers(0,2,size=(args.batch_size, args.k*args.block_len, 1)).astype(np.float32)
            loss_v, acc_v = val_step(b)
  #          print(f' step {step+1}: {loss_v:.3f}', end='\r')
        val_history['loss'].append(loss_v)
        val_history['binary_accuracy'].append(acc_v)
        print(f' validation: loss {loss_v:.4f}, acc {acc_v:.4f}')


        # reset metrics
        acc_fn_e.reset_states()
        acc_fn_d.reset_states()
        acc_fn_v.reset_states()

        # perform callbacks

        if not is_progress(val_history, last_check_point):
            if updates < 3:
                # args.batch_size *= 2
                args.learning_rate /= 2
                # print(f'\nincreasing batch size to {args.batch_size} and reducing learning rate to {args.learning_rate}')
                print(f'\nreducing learning rate to {args.learning_rate}')
                updates +=1
                last_check_point = epoch
            else:
                print(f'\nstopping early')
                break
        print('')

    ###
    # save
    ###
    
    # the model
    print('saving model...')
    model.save_weights(os.path.join(args.test_dir, f'{args.experiment}_{args.test}_weights'))

    # training 
    print('saving history...')
    np.save(os.path.join(args.test_dir, f'encoder_history.npy'), encoder_history, allow_pickle=True)
    np.save(os.path.join(args.test_dir, f'decoder_history.npy'), decoder_history, allow_pickle=True)
    np.save(os.path.join(args.test_dir, f'val_history.npy'), val_history, allow_pickle=True)



    ###
    # test over various block length
    ###
    # error

    # bit error location

    # encoder min distance

    # encoder power distribution

if __name__ == '__main__':
    parser = ArgumentParser(description='control script for single user arbitrary rate nn code experiment')

    # base args
    parser.add_argument('n', type=int, help='num symbols')
    parser.add_argument('snr', type=float, help='signal to noise ratio in dB')
    parser.add_argument('-L', '--block_len', type=int, default=100, help='training block length')
    parser.add_argument('-k', type=int, default=None, help='num bits')
    
    parser.add_argument('--encoder_layers', type=int, help='number of layers in encoder')
    parser.add_argument('--encoder_filters', type=int, help='number of filters per layer in encoder')
    parser.add_argument('--encoder_kernel_size', type=int, help='kernel size for the conv1d layers in encoder')
    parser.add_argument('--encoder_dilation_rate', type=int, help='dilation rate for the conv1d layers in encoder')

    parser.add_argument('--decoder_layers', type=int, help='number of layers in decoder')
    parser.add_argument('--decoder_filters', type=int, help='number of filters per layer in decoder')
    parser.add_argument('--decoder_kernel_size', type=int, help='kernel size for the conv1d layers in decoder')
    parser.add_argument('--decoder_dilation_rate', type=int, help='dilation rate for the conv1d layers in decoder')


    parser.add_argument('--train_steps', type=int, help='number of training steps per epoch')
    parser.add_argument('--encoder_steps', type=int, help='number of encoder training steps per epoch')
    parser.add_argument('--decoder_steps', type=int, help='number of decoder training steps per epoch')
    parser.add_argument('--val_steps', type=int, help='number of validation steps per epoch')
    parser.add_argument('--learning_rate', type=float, help='learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    # experiment args
    parser.add_argument('--experiment', type=str, help='experiment name')
    parser.add_argument('--test', type=int, help='test number within experiment')

    args = parser.parse_args()

    print('parsed args:')
    print(args)
    # make an experiment and test folder
    test_dir = os.path.join(os.getenv('BASEDIR'), f'experiment_{args.experiment}', f'test_{args.test}')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    args.test_dir = test_dir
   
    with open(os.path.join(test_dir, 'args.json'), 'w') as f:
        print('writing args to file...')
        json.dump(vars(args), f)
    
    print('calling "run_experiment"')
    run_experiment(args)
