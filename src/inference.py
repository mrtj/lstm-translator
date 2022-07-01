from pathlib import Path

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

from utils import preproc, load_vocab, load_meta
from train import TrainableModel

class Predictor:
    
    def __init__(self, latent_dim, vocab_dir, weights_file):
        self.vocab_dir = Path(vocab_dir)
        self.latent_dim = latent_dim
        self.max_length_src, _ = load_meta(self.vocab_dir / 'meta.json')
        self.input_token_index, _ = load_vocab(self.vocab_dir / 'source.txt')
        self.target_token_index, self.reverse_target_index = load_vocab(self.vocab_dir / 'target.txt')
        num_encoder_tokens = len(self.input_token_index)
        num_decoder_tokens = len(self.target_token_index)
        num_decoder_tokens += 1 # For zero padding
        
        self.trainable_model = TrainableModel(num_encoder_tokens, num_decoder_tokens, latent_dim)
        self.trainable_model.load_state(weights_file)
        
        self.encoder_model = Predictor.create_encoder_model(
            self.trainable_model.encoder_inputs, 
            self.trainable_model.encoder_states
        )

        self.decoder_model = Predictor.create_decoder_model(
            self.trainable_model.decoder_inputs,
            self.trainable_model.decoder_embedding_layer,
            self.trainable_model.decoder_lstm, 
            self.trainable_model.decoder_dense, 
            latent_dim
        )
        
    def predict(self, text):
        tokens = Predictor.tokenize_sentence(text, self.max_length_src, self.input_token_index)
        translated = self.translate_sequence(tokens)
        return translated[:-len('_END')].strip()

    def translate_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = self.target_token_index['START_']

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_index[sampled_token_index]
            decoded_sentence += ' ' + sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '_END' or
               len(decoded_sentence) > 50):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence

    @staticmethod
    def create_encoder_model(encoder_inputs, encoder_states):
        return Model(encoder_inputs, encoder_states)

    @staticmethod
    def create_decoder_model(decoder_inputs, decoder_embedding_layer, decoder_lstm, decoder_dense, latent_dim=50):
        decoder_state_input_h = Input(shape=(latent_dim,), name='encoder_hidden_state')
        decoder_state_input_c = Input(shape=(latent_dim,), name='encoder_cell_state')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        dec_emb2 = decoder_embedding_layer(decoder_inputs) # Get the embeddings of the decoder sequence

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

        # Final decoder model
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs2] + decoder_states2)
        return decoder_model

    @staticmethod
    def tokenize_sentence(text, max_length_src, input_token_index):
        encoded = np.zeros((max_length_src), dtype='float32')
        text = preproc(text)
        for t, word in enumerate(text.split()):
            encoded[t] = input_token_index[word] # encoder input seq
        return np.expand_dims(encoded, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str)
    parser.add_argument('--latent-dim', type=int, default=50)
    parser.add_argument('--vocab-dir', type=str, default=os.environ.get('SM_CHANNEL_VOCAB'))

    args, _ = parser.parse_known_args()
    predictor = Predictor(args.latent_dim, args.vocab_dir)
    predictor.predict(args.text)

