from __future__ import print_function
import numpy as np
from flask import abort, current_app, g
from flask_restful import reqparse
from flask_restful_swagger_2 import swagger, Resource
from app.resources.schemas import Entity, ParseResponse, ParseBatchResponse
from app.utils import get_model
from utils import create_input
from loader import prepare_dataset

class Parse(Resource):
    """
    """
    parser = reqparse.RequestParser()
    parser.add_argument('string', type=unicode, trim=True, required=True, location='json')
    @swagger.doc({
        'description': 'Parse a single string and return the associated entity for each token in the string.',
        'reqparser': {
            'name': 'Single Submission Request',
            'parser': parser
        },
        'responses': {
            '200': {
                'description': 'Successfully parsed provided string.',
                'schema': ParseResponse
            }
        }
    })

    def post(self):
        """
        Parse a single string and return the associated entity for each token in the string.
        """
        args = self.parser.parse_args()
        ref_string = args.get('string')
        if ref_string is None or ref_string == "":
            # Hackish way as reqparse can't catch empty string
            abort(400, description='string is empty or not provided.')

        tokens = ref_string.split(" ")

        data = prepare_dataset([[[token] for token in tokens]],
                               current_app.word_to_id,
                               current_app.char_to_id,
                               {},
                               current_app.model.parameters['lower'],
                               True)

        model_inputs = create_input(data[0], current_app.model.parameters, False)
        y_pred = np.array(current_app.inference[1](*model_inputs))[1:-1]
        tags = [current_app.model.id_to_tag[y_pred[i]] for i in range(len(y_pred))]

        response = ParseResponse(reference_string=ref_string,
                                 data=[Entity(term=term, entity=entity)
                                       for term, entity in zip(tokens, tags)])
        return response

class ParseBatch(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('strings', type=unicode, action='append', required=True, location='json')
    @swagger.doc({
        'description': 'Parse multiple string and return the associated entity for each token in each string.',
        'reqparser': {
            'name': 'Mutliple Submission Request',
            'parser': parser
        },
        'responses': {
            '200': {
                'description': 'Successfully parsed provided strings.',
                'schema': ParseBatchResponse
            }
        }
    })
    def post(self):
        """
        Parse multiple string and return the associated entity for each token in each string.
        """
        args = self.parser.parse_args()
        ref_strings = args.get('strings')

        tokens = [[[token] for token in ref_string.split(" ")] for ref_string in ref_strings]
        data = prepare_dataset(tokens,
                               current_app.word_to_id,
                               current_app.char_to_id,
                               {},
                               current_app.model.parameters['lower'],
                               True)

        tagged = []

        for index, datum in enumerate(data):
            model_inputs = create_input(datum, current_app.model.parameters, False)
            y_pred = np.array(current_app.inference[1](*model_inputs))[1:-1]
            tags = [current_app.model.id_to_tag[y_pred[i]] for i in range(len(y_pred))]

            tagged.append([Entity(term=term, entity=entity)
                           for term, entity in zip(ref_strings[index].split(" "), tags)])

        response = ParseBatchResponse(reference_strings=ref_strings,
                                      data=tagged)
        return response
