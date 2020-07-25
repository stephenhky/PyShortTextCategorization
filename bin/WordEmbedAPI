#!/usr/bin/env python

# argument parsing
import argparse

argparser = argparse.ArgumentParser(description='Load word-embedding models into memory.')
argparser.add_argument('filepath', help='file path of the word-embedding model')
argparser.add_argument('--port', type=int, default=5000, help='port number')
argparser.add_argument('--embedtype', default='word2vec', help='type of word-embedding algorithm (default: "word2vec), allowing "word2vec", "fasttext", and "poincare"')
argparser.add_argument('--debug', default=False, action='store_true', help='Debug mode (Default: False)')
args = argparser.parse_args()


from flask import Flask, request, jsonify
import shorttext

app = Flask(__name__)
if args.embedtype == 'word2vec':
    w2v_model = shorttext.utils.load_word2vec_model(args.filepath, binary=True)
elif args.embedtype == 'fasttext':
    w2v_model = shorttext.utils.load_fasttext_model(args.filepath)
elif args.embedtype == 'poincare':
    w2v_model = shorttext.utils.load_poincare_model(args.filepath, binary=True)
else:
    raise KeyError("Argument 'embedtype' {} unknown.".format(args.embedtype))


@app.route('/closerthan',methods=['POST'])
def closer_than():
    data = request.get_json(force=True)
    entity1 = data['entity1']
    entity2 = data['entity2']
    close_entities = w2v_model.closer_than(entity1, entity2)
    return jsonify(close_entities)


@app.route('/distance',methods=['POST'])
def distance():
    data = request.get_json(force=True)
    entity1 = data['entity1']
    entity2 = data['entity2']
    distance = w2v_model.distance(entity1, entity2)
    return jsonify({'distance': distance})


@app.route('/distances',methods=['POST'])
def distances():
    data = request.get_json(force=True)
    entity1 = data['entity1']
    other_entities = tuple(data['other_entities'])
    distances = w2v_model.distances(entity1, other_entities)
    return jsonify({'distances': list([float(distance) for distance in distances])})


@app.route('/get_vector',methods=['POST'])
def get_vector():
    data = request.get_json(force=True)
    token = data['token']
    try:
        vector = w2v_model.get_vector(token)
        return jsonify({'vector': vector.tolist()})
    except KeyError:
        return jsonify({})


@app.route('/most_similar',methods=['POST'])
def most_similar():
    keyword_args = request.get_json(force=True)
    returned_results = w2v_model.most_similar(**keyword_args)
    return jsonify(returned_results)


@app.route('/most_similar_to_given',methods=['POST'])
def most_similar_to_given():
    data = request.get_json(force=True)
    entity1 = data['entity1']
    entities_list = data['entities_list']
    entity = w2v_model.most_similar_to_given(entity1, entities_list)
    return jsonify({'token': entity})


@app.route('/rank',methods=['POST'])
def rank():
    data = request.get_json(force=True)
    entity1 = data['entity1']
    entity2 = data['entity2']
    rank = w2v_model.rank(entity1, entity2)
    return jsonify({'rank': rank})


@app.route('/similarity',methods=['POST'])
def similarity():
    data = request.get_json(force=True)
    entity1 = data['entity1']
    entity2 = data['entity2']
    similarity = w2v_model.similarity(entity1, entity2)
    return jsonify({'similarity': float(similarity)})


if __name__ == "__main__":
    app.run(debug=args.debug, port=args.port)