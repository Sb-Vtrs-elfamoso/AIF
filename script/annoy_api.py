from flask import Flask, request, jsonify
from annoy import AnnoyIndex

app = Flask(__name__)

# Load the Annoy database
annoy_db = AnnoyIndex(576, metric='angular')  # the dimension of the vectors in the database
annoy_db.load('./annoy_db.ann')

@app.route('/') # This is the home route, it just returns 'Hello world!'
def index():    # I use it to check that the server is running and accessible it's not necessary
    return 'Hello world!'

@app.route('/reco', methods=['POST']) # This route is used to get recommendations
def reco():
    vector = request.json['vector'][0] # Get the vector from the request
    closest_indices = annoy_db.get_nns_by_vector(vector, 6) # Get the 5 closest elements indices
    reco = [closest_indices[1], closest_indices[2], closest_indices[3], closest_indices[4], closest_indices[5]]  # Assuming the indices are integers
    return jsonify(reco) # Return the reco as a JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True) # Run the server on port 5000 and make it accessible externally
