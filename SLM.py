import requests
import json
import os
from flask import Flask, request
import pandas as pd
#import numpy as np
from SLMHub import Output
from sklearn.externals import joblib
from .SLMOtak import SLMOtak

app = Flask(__name__)

class KnowledgeGraph:
    def __init__(self, data):
        self.data = data  # store the data
        self.graph = self.create_graph()  # create the graph when the instance is created
        self.SLMOtak = SLMOtak()

    def create_graph(self):
        # This is where you would add the code to create a knowledge graph from self.data
        # For now, let's just return an empty graph
        return {}
    
class SLMProcessor:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.api_endpoint = config['api_endpoint']
        self.training_dataset_filename = config['training_dataset_filename']

    def load_training_dataset(self, filename):
        # Build the file path
        file_path = os.path.join(os.getcwd(), filename)

        # Load the training dataset
        self.training_dataset = pd.read_csv(file_path)
        return self.training_dataset
       
    def get_slm_output(self, data):
        try:
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
        
            # Extract the combined_output, nmf_topics, topics_assignments, and topics_models from the response
            combined_output = data.get('combined_output')
            nmf_topics = data.get('nmf_topics')
            topics_assignments = data.get('topics_assignments')
            topics_models = data.get('topics_models')

            return combined_output, nmf_topics, topics_assignments, topics_models

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"An error occurred: {err}")

        return None, None, None, None

    def uncombine_output(self, combined_output, nmf_topics, topics_assignments, topics_models):
        # Convert combined_output from JSON to a Python dictionary
        combined_output = json.loads(combined_output)

        # Convert nmf_topics, topics_assignments, and topics_models from JSON to Pandas DataFrames
        nmf_topics = pd.read_json(nmf_topics)
        topics_assignments = pd.read_json(topics_assignments)
        topics_models = pd.read_json(topics_models)

        # Extract the modalities from the combined_output
        text_data = combined_output.get('text')
        image_data = combined_output.get('image')
        video_data = combined_output.get('video')
        audio_data = combined_output.get('audio')

        # Return the uncombined output as a dictionary
        return {
            'text': text_data,
            'image': image_data,
            'video': video_data,
            'audio': audio_data,
            'nmf_topics': nmf_topics,
            'topics_assignments': topics_assignments,
            'topics_models': topics_models
        }
    
    # process the data for each modality as necessary
    def identify_and_process_modality(self, output):
        processed_output = []
        rejected_modalities = []

        for modality in ['text', 'image', 'video', 'audio']:
            if output.get(modality):
                if self.can_process(modality):
                    processed_output.append(self.process_modality(modality, output.get(modality)))
                else:
                    rejected_modalities.append(modality)

        return processed_output, rejected_modalities

    def can_process(self, modality, data):
        # Load the trained model
        model = joblib.load('model.pkl')

        # Predict the modality of the data
        predicted_modality = model.predict([data])

        # Get the probability of the predicted modality
        confidence_score = model.predict_proba([data])[0][predicted_modality]

        # Check if the predicted modality matches the given modality
        if predicted_modality == modality:
            if confidence_score > 0.7:
                # If the confidence score is above 0.7, process the modality
                return True
            else:
                # If the confidence score is 0.7 or below, reject the modality
                return False
        else:
            return False 
           
    def process_modality(self, modality, data):
         # Create a knowledge graph instance
        kg = KnowledgeGraph(data)  # replace with actual code to create a knowledge graph
        
        # Save the knowledge graph data
        with open('kg_data.json', 'w') as f:
            json.dump({'modality': modality, 'data': kg.data}, f)  # replace 'kg.data' with the actual data you want to save
        return data        
    
    def send_rejected_modalities(self, rejected_modalities):
        try:
            response = requests.post('https://API_ENDPOINT', json={'rejected_modalities': rejected_modalities})  # replace 'API_ENDPOINT' with the actual API endpoint
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"An error occurred: {err}")

    def process_slm_output(self):
        output = Output(processed_output, rejected_modalities)  # create an instance of Output
        if output is None:
            print('No output to process.')
            return
        processed_output, rejected_modalities = self.identify_and_process_modality(output)
        if rejected_modalities:
            print('Modalities rejected, model not confident.')
            return
        # process the output as necessary
        # handle rejected modalities as necessary

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    try:
        filename = request.form['filename']
        slm = SLMProcessor()
        slm.load_training_dataset(filename)
        return 'Dataset loaded successfully', 200
    except FileNotFoundError:
        return f'File {filename} not found', 400
    except pd.errors.ParserError:
        return f'Unable to parse file {filename}', 400
    except Exception as e:
        return f'An error occurred: {str(e)}', 500
    
if __name__ == "__main__":
    app.run(debug=True)    