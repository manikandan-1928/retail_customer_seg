from flask import Flask, request, jsonify,render_template

app1 = Flask(__name__)

@app1.route('/')
def home():
    return render_template('home.html')


@app1.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})
    
    file = request.files['file']
    # Process the file and do necessary operations
    
    # Return a response
    return jsonify({'message': 'File processed successfully'})

if __name__ == "__main__":
    app1.run(debug=True)
