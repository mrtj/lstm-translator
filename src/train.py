import argparse, os
from pathlib import Path
import json

import numpy as np
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import load_vocab, load_meta

class TrainableModel:
    ''' Creates a trainable version of the model and saves some layers. '''
    
    def __init__(self, num_encoder_tokens, num_decoder_tokens, latent_dim=50):        
        self.latent_dim = latent_dim

        encoder_inputs = Input(shape=(None,), name='encoder_input')
        enc_emb = Embedding(num_encoder_tokens, latent_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
        encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        self.encoder_inputs = encoder_inputs
        self.encoder_states = encoder_states
        
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,), name='decoder_input')
        dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero=True, name='decoder_embedding')
        dec_emb = dec_emb_layer(decoder_inputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.decoder_inputs = decoder_inputs
        self.decoder_dense = decoder_dense
        self.decoder_lstm = decoder_lstm
        self.decoder_embedding_layer = dec_emb_layer

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='trainable_model')
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
                
        self.model = model

    def load_state(self, weights_filename):
        self.model.load_weights(weights_filename)

    def save_state(self, weights_filename):
        model.save_weights(weights_filename)


def generate_batch(
    X, y, 
    max_length_src, max_length_tar, 
    num_decoder_tokens, 
    input_token_index, target_token_index, 
    batch_size=128
):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar), dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                for t, word in enumerate(target_text.split()):
                    if t<len(target_text.split())-1:
                        decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data], decoder_target_data)


def train(args):
    epochs     = args.epochs
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    verbose    = args.verbose
    model_save_dir = Path(args.model_save_dir)
    train_dir  = Path(args.train_dir)
    test_dir   = Path(args.test_dir)
    vocab_dir  = Path(args.vocab_dir)
    
    print(f'Resolved training args: {args}')
    
    # load vocabs
    input_token_index, reverse_input_index = load_vocab(vocab_dir / 'source.txt')
    target_token_index, reverse_target_index = load_vocab(vocab_dir / 'target.txt')
    num_encoder_tokens = len(input_token_index)
    num_decoder_tokens = len(target_token_index)
    num_decoder_tokens += 1 # For zero padding
    
    print(f'num_encoder_tokens={num_encoder_tokens}, num_decoder_tokens={num_decoder_tokens}')
    
    max_length_src, max_length_tar = load_meta(vocab_dir / 'meta.json')
    
    print(f'max_length_src={max_length_src}, max_length_target={max_length_tar}')
    
    # load dataset
    X_train = np.load(train_dir / 'x.npy', allow_pickle=True)
    y_train = np.load(train_dir / 'y.npy', allow_pickle=True)
    X_test = np.load(test_dir / 'x.npy', allow_pickle=True)
    y_test = np.load(test_dir / 'y.npy', allow_pickle=True)

    train_generator = generate_batch(
        X_train, y_train, 
        max_length_src, max_length_tar,
        num_decoder_tokens,
        input_token_index, target_token_index,
        batch_size=batch_size
    )
    
    val_generator = generate_batch(
        X_test, y_test,
        max_length_src, max_length_tar,
        num_decoder_tokens,
        input_token_index, target_token_index,
        batch_size=batch_size
    )
    
    train_samples = len(X_train)
    val_samples = len(X_test)
    
    # create model
    trainable = TrainableModel(num_encoder_tokens, num_decoder_tokens, latent_dim)
    
    # dataset saving
    model_checkpoint_callback = ModelCheckpoint(
        filepath=str(model_save_dir / 'weights-{epoch:02d}.hdf5'),
        save_weights_only=True,
        monitor='acc',
        mode='max',
        save_best_only=True
    )
    
    train_steps = train_samples // batch_size
    val_steps = val_samples // batch_size
    
    print(f'train_steps={train_steps}, val_steps={val_steps}')
    
    # kick off training
    trainable.model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=[model_checkpoint_callback],
        verbose=verbose
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=50)
    parser.add_argument('--verbose', type=int, default=2, 
                        help='Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')
    parser.add_argument('--model-save-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test-dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--vocab-dir', type=str, default=os.environ.get('SM_CHANNEL_VOCAB'))

    args, _ = parser.parse_known_args()
    train(args)
