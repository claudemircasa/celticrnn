""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
import argparse
import tensorflow as tf
from tqdm import tqdm
from random import randint, sample
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint

EPOCHS = 500
BATCHS = 512
SEQUENCE = 64

OPTIMIZER = 'adagrad'
LOSS = 'categorical_crossentropy'

parser = argparse.ArgumentParser(description='train celticrnn network.')
parser.add_argument('--fdump', type=int, default=0, help='force generation of notes dump file')
parser.add_argument('--mfile', type=int, help='number of files of dataset to use')
parser.add_argument('--ngpus', type=int, default=1, help='number of gpus to use')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu device')
parser.add_argument('-m', '--weights', type=str, help='checkpoint to resume')
args = parser.parse_args()

def train_network():
    """ Train a Neural Network to generate music """
    notes = None
    if (args.fdump == 1):
        notes = get_notes()
    else:
        with open('data/notes', 'rb') as filepath:
            notes = pickle.load(filepath)

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    if (args.mfile):
        files = tqdm(sample(glob.glob('dataset/*.mid'), args.mfile))
    else:
        files = tqdm(glob.glob('dataset/*.mid'))

    for file in files:
        midi = converter.parse(file)

        files.set_description("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            # notes_to_parse = s2.parts[0]
            notes_to_parse = s2.parts
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for _instrument in notes_to_parse:
            # the first element is the instrument midi representation
            ri = instrument.Instrument.__subclasses__()
            # iid = ri[randint(0, len(ri)-1)]().instrumentName.replace(' ', '_')
            iid = ri[randint(0, len(ri)-1)]().midiProgram

            # format is: [<instrument>, <note>, <duration>]
            if (isinstance(_instrument, note.Note)):
                notes.append('%s %s %s %s' % (iid, str(_instrument.pitch), _instrument.duration.quarterLength, _instrument.offset))
            elif (isinstance(_instrument, stream.Part)):
                if (not _instrument.getInstrument(returnDefault=False).instrumentName == None):
                    iid = _instrument.getInstrument(returnDefault=False).midiProgram
                for element in _instrument:
                    if isinstance(element, note.Note):
                        notes.append('%s %s %s %s' % (iid, str(element.pitch), element.duration.quarterLength, element.offset))
                    elif isinstance(element, chord.Chord):
                        notes.append('%s %s %s %s' % (iid, ' '.join(str(p) for p in element.pitches), element.duration.quarterLength, element.offset))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = SEQUENCE

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    nmodel = None
    if (args.weights):
        nmodel = load_model(args.weights)
    else:
        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            recurrent_dropout=0.3,
            return_sequences=True,
        ))
        model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.2,))
        model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.1,))
        model.add(LSTM(256))
        model.add(BatchNorm())
        model.add(Dropout(0.3))
        model.add(Dense(256))
        model.add(Activation('tanh'))
        model.add(BatchNorm())
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))

        if (args.ngpus > 1):
            print('INFO: using %d devices' % args.ngpus)
            parallel_model = multi_gpu_model(model, gpus=2)
            parallel_model.compile(loss=LOSS, optimizer=OPTIMIZER)
            nmodel = parallel_model
        else:
            print('INFO: using only one device')
            model.compile(loss=LOSS, optimizer=OPTIMIZER)
            nmodel = model
    return nmodel

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCHS, callbacks=callbacks_list)

if __name__ == '__main__':
    with tf.device('/gpu:%d' % args.gpu):
        print('INFO: using gpu at index %d' % args.gpu)
        train_network()
