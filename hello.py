'''
Hi, this is a pythonic API for IA services.
Here we can process data with IA and serve it to the world.
We will use Scikit Learn to server IA services for our 
ecosystem.
'''
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return jsonify({'message' : 'Hello World'})

if __name__ == '__main__':
    ''' Starts a server on the port 5000 '''
    app.run()
