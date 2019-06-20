import tensorflow as tf
import sys
import itertools
import joblib

# function required specificly for this keras model
def build_model():
    return

# function required specificly for this keras model
def build_model_soh():
    return

# path for checkpints
model_path = './checkpoints/'

#path for c code dir
c_path = './c_process/'

def main():
    # checks if model name was provided
    if len(sys.argv) == 2:
        model = sys.argv[1]
    else:
        print("Too few arguments!")
        return

    # loads sklearn pipeline
    # if you only use keras, load just keras weights
    pipeline = joblib.load('{}{}.pkl'.format(model_path, model))
    pipeline.named_steps['keras'].model = tf.keras.models.load_model('{}{}.h5'.format(model_path, model))
    
    # writes scaler's variables to file
    f = open('{}/scaler.txt'.format(c_path), "w")
    f.write(' '.join(str(e) for e in pipeline.named_steps['scaler'].mean_))
    f.write('\n')
    f.write(' '.join(str(e) for e in pipeline.named_steps['scaler'].scale_))
    f.write('\n')

    wh = pipeline.named_steps['keras'].model.get_weights()
    f = open('{}layers.txt'.format(c_path), "w")

    # writes weights of keras model to file
    # also writes two numbers
    # input - count of inputs / last layer's node count
    # unit - count of next layer's node count
    for i in range(0, len(wh), 2):
        inp = len(wh[i])
        uni = len(wh[i + 1])
        f.write("{} {}\n".format(inp, uni))
        merged = list(itertools.chain.from_iterable(wh[i]))
        f.write(' '.join(str(e) for e in merged))
        f.write('\n')
        f.write(' '.join(str(e) for e in wh[i + 1]))
        f.write('\n')

if __name__ == "__main__":
    main()
