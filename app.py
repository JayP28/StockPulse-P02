import json
import os
from flask import Flask
from flask_cors import CORS
from models import db, Episode, Review
from routes import register_routes

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# Configure SQLite database - using 3 slashes for relative path
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)

# Function to initialize database, change this to your own database initialization logic
def init_db():
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Initialize database with data from init.json if empty
        if Episode.query.count() == 0:
            json_file_path = os.path.join(current_directory, 'init.json')
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                for episode_data in data['episodes']:
                    episode = Episode(
                        id=episode_data['id'],
                        title=episode_data['title'],
                        descr=episode_data['descr']
                    )
                    db.session.add(episode)
                
                for review_data in data['reviews']:
                    review = Review(
                        id=review_data['id'],
                        imdb_rating=review_data['imdb_rating']
                    )
                    db.session.add(review)
            
            db.session.commit()
            print("Database initialized with episodes and reviews data")

init_db()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)