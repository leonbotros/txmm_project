from keras.models import load_model
import json

def get_dev_data(dev_data):
    return NotImplementedError

def main():
    dev_data = json.load(open("data/dev-v1.1.json"))['data']
    model = load_model('simple_bidaf.h5')

    # ps_start, ps_end = model.predict([,])
    # for s, e in zip(ps_start, ps_end):
    #     max_s = np.argmax(s)
    #     max_e = np.argmax(e)
    #     print(max_s, max_e)



if __name__ == "__main__":
    main()