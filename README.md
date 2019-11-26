# Celtic Music Composer

This project allows you to train a neural network to generate midi music files that make use of a single instrument

## Requirements

* Python 3.x
* Use pip to install requirements:

```bash
pip install -r requirements.txt
```

## Training

To train the network you run **lstm.py**.

E.g.

```
python lstm.py
```

The network will use every midi file in ./midi_songs to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

## Generating music

Once you have trained the network you can generate text using **predict.py**

E.g.

```
python predict.py
```

You can run the prediction file right away using the **weights.hdf5** file

Please cite this work and the original work:

```
@misc{celticrnn,
 title   ={CelticRNN: Uma rede neural recorrente para a geração de música Celta},
 url     ={https://github.com/claudemircasa/celticrnn/},
 organization={IMAGO Research Group},
 urlaccessdate={26 nov. 2019}
}
```
