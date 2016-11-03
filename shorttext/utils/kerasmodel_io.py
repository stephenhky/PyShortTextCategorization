from keras.models import model_from_json

def save_model(nameprefix, model):
    model_json = model.to_json()
    open(nameprefix+'.json', 'wb').write(model_json)
    model.save_weights(nameprefix+'.h5')

def load_model(nameprefix):
    model = model_from_json(open(nameprefix+'.json', 'rb').read())
    model.load_weights(nameprefix+'.h5')
    return model