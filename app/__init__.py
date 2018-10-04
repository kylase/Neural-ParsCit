import os
import logging
from flask import Flask, Blueprint, jsonify, g
from flask_restful_swagger_2 import Api, get_swagger_blueprint
from flask_swagger_ui import get_swaggerui_blueprint
from app.resources.parscit import Parse, ParseBatch
from .utils import get_model


def create_app(config):
    """
    Wrapper function for Flask app
    params:
        config: Config
    """
    app = Flask(__name__)
    app.config.from_object(config)

    model_path = os.path.abspath(os.getenv('MODEL_PATH',
                                           default='models/neuralParscit/'))
    word_emb_path = os.path.abspath(os.getenv('WORD_EMB_PATH',
                                              default='vectors_with_unk.kv'))

    with app.app_context():
        logging.info("Loading model from {} and using word embeddings from {}".format(model_path, word_emb_path))
        model, inference = get_model(model_path, word_emb_path)
        setattr(app, 'model', model)
        setattr(app, 'inference', inference)
        setattr(app, 'word_to_id', {v:i for i, v in model.id_to_word.items()})
        setattr(app, 'char_to_id', {v:i for i, v in model.id_to_char.items()})

    API_DOC_PATH = '/docs'
    SWAGGER_PATH = '/swagger'

    api_bp = Blueprint('api', __name__)
    api = Api(api_bp, add_api_spec_resource=False)
    api.add_resource(Parse, '/parscit/parse')
    api.add_resource(ParseBatch, '/parscit/parse/batch')

    docs = [api.get_swagger_doc()]

    swagger_ui_blueprint = get_swaggerui_blueprint(
        API_DOC_PATH,
        SWAGGER_PATH + '.json',
        config={
            'app_name': 'ParsCit API'
        }
    )

    app.register_blueprint(api.blueprint)
    app.register_blueprint(get_swagger_blueprint(docs, SWAGGER_PATH,
                                                 title='ParsCit API',
                                                 api_version='1.0',
                                                 base_path='/'))
    app.register_blueprint(swagger_ui_blueprint, url_prefix=API_DOC_PATH)

    @app.errorhandler(404)
    def not_found(error):
        """
        Handles URLs that are not specified
        """
        return jsonify({
            'message': "API doesn't exist"
        }), 404

    return app
