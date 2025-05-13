import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Function to load and preprocess the dataset
def preprocess_spotify_data(file_path='spotify-2023.csv'):
    # Load the dataset
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Display basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values[missing_values > 0])
    
    # Check data types of columns and handle any problematic columns
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
        
        # For 'bpm' column, check if it contains string values
        if col == 'bpm':
            # Check if there are any non-numeric values
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna() & ~df[col].isna()
            if non_numeric.any():
                print(f"Non-numeric values found in '{col}' column. Example: {df.loc[non_numeric, col].iloc[0]}")
                
                # Try to extract numeric values from strings if possible
                try:
                    # Extract numbers using regex pattern
                    import re
                    def extract_number(value):
                        if isinstance(value, str) and 'BPM' in value:
                            # Extract the numeric value after 'BPM'
                            match = re.search(r'BPM(\d+)', value)
                            if match:
                                return float(match.group(1))
                        return value
                    
                    # Apply the extraction function
                    df[col] = df[col].apply(extract_number)
                    
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"Fixed '{col}' column by extracting numeric values.")
                except Exception as e:
                    print(f"Error processing '{col}' column: {str(e)}")
                    # If extraction fails, set problematic values to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # For numerical features, fill with median
    numerical_cols = ['artist_count', 'released_year', 'released_month', 'released_day',
                      'in_spotify_playlists', 'in_spotify_charts', 'streams', 
                      'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists',
                      'in_deezer_charts', 'in_shazam_charts', 'bpm', 
                      'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
                      'instrumentalness_%', 'liveness_%', 'speechiness_%']
    
    for col in numerical_cols:
        if col in df.columns:
            # Check if column can be treated as numeric
            try:
                # Convert to numeric first to ensure we can calculate median
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")
                # If errors occur, use the mode instead
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
    
    # For categorical features, fill with mode
    categorical_cols = ['track_name', 'artist(s)_name', 'key', 'mode']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
    
    # Clean and transform streams data if it exists
    if 'streams' in df.columns:
        # Convert streams to numeric (removing commas if present)
        try:
            df['streams'] = df['streams'].astype(str).str.replace(',', '')
            df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
        except Exception as e:
            print(f"Error cleaning streams column: {str(e)}")
    
    # Normalize numerical features for better performance in GNN
    print("Normalizing numerical features...")
    scaler = MinMaxScaler()
    features_to_normalize = [
        'in_spotify_playlists', 'in_spotify_charts', 'streams', 
        'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists',
        'in_deezer_charts', 'in_shazam_charts', 'bpm', 
        'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
        'instrumentalness_%', 'liveness_%', 'speechiness_%'
    ]
    
    # Only normalize columns that exist in the dataset
    features_to_normalize = [col for col in features_to_normalize if col in df.columns]
    
    if features_to_normalize:
        df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize].fillna(0))
        print(f"Normalized features: {features_to_normalize}")
    
    # Create an ID column for each track
    print("Creating track_id column...")
    df['track_id'] = range(len(df))
    
    return df

# Function to create graph structure for GNN
def create_graph_structure(df):
    """
    Create a graph structure from the preprocessed DataFrame.
    """
    print("Creating graph structure...")
    G = nx.Graph()
    
    # Create a unique track_id if it doesn't exist
    if 'track_id' not in df.columns:
        print("'track_id' column not found. Creating unique IDs based on track_name and artist...")
        # Create a unique identifier using track_name and artist(s)_name
        df['track_id'] = df.apply(lambda row: f"{row['track_name']}_{row['artist(s)_name']}".replace(' ', '_'), axis=1)
    
    # Add song nodes
    for idx, row in df.iterrows():
        song_attrs = {
            'name': row['track_name'],
            'artist': row['artist(s)_name'],
            'year': row['released_year'],
            'streams': row['streams'] if 'streams' in df.columns else 0,
            'type': 'song'
        }
        
        # Add additional attributes if they exist
        for feature in ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                       'instrumentalness_%', 'liveness_%', 'speechiness_%']:
            if feature in df.columns:
                song_attrs[feature.replace('%', '')] = row[feature]
        
        G.add_node(f"song_{row['track_id']}", **song_attrs)
    
    # Add artist nodes and edges between songs and artists
    artists = set()
    for idx, row in df.iterrows():
        # Split artist names if there are multiple artists
        artist_list = row['artist(s)_name'].split(', ')
        
        for artist in artist_list:
            # Add artist node if it doesn't exist
            if artist not in artists:
                G.add_node(f"artist_{artist}", name=artist, type='artist')
                artists.add(artist)
            
            # Add edge between song and artist
            G.add_edge(f"song_{row['track_id']}", f"artist_{artist}")
    
    # Create edges between songs based on feature similarity
    # Extract song features for similarity calculation
    features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
    
    # Only use features that exist in the dataset
    features = [f for f in features if f in df.columns]
    
    if features:
        print(f"Computing song similarities based on: {features}")
        # Handle non-numeric features
        feature_df = df[features].copy()
        
        threshold = 0.75  # Changed from 0.9 to 0.75 to create more connections
        print(f"Using similarity threshold: {threshold}")
        
        # Process in chunks to avoid memory issues
        from sklearn.metrics.pairwise import cosine_similarity
        
        chunk_size = 100  # Adjust based on memory constraints
        for i in range(0, len(df), chunk_size):
            end_idx = min(i + chunk_size, len(df))
            chunk = feature_df.iloc[i:end_idx]
            
            # Calculate similarities with all other songs
            similarities = cosine_similarity(chunk, feature_df)
            
            for j in range(len(chunk)):
                song_idx = i + j
                song_id = f"song_{df.iloc[song_idx]['track_id']}"
                
                for k in range(len(similarities[j])):
                    if song_idx != k and similarities[j][k] > threshold:
                        other_song_id = f"song_{df.iloc[k]['track_id']}"
                        G.add_edge(song_id, other_song_id, weight=similarities[j][k])
    
    print(f"Graph created with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

# Function to prepare final features for GNN
def prepare_gnn_features(df, G):
    """
    Prepare node and edge features for GNN models.
    """
    print("Preparing features for GNN...")
    node_features = {}
    edge_features = {}
    
    # Get numerical features from the dataframe
    numerical_features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                         'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
    
    # Only use features that exist in the dataset
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    # Create a lookup dictionary for node features
    track_features = {}
    for idx, row in df.iterrows():
        track_id = row['track_id']
        features = []
        
        # Add numerical features
        for feature in numerical_features:
            # Convert to numeric and handle NaN values
            value = pd.to_numeric(row[feature], errors='coerce')
            features.append(float(value) if not pd.isna(value) else 0.0)
        
        # Store features for this track
        track_features[track_id] = np.array(features, dtype=np.float32)
    
    # Dictionary to store average features for artists
    artist_avg_features = {}
    
    # First pass: collect all songs for each artist
    artist_songs = {}
    for node in G.nodes():
        if node.startswith('song_'):
            # Get the song's artist(s)
            track_id = int(node[5:])  # Remove 'song_' prefix
            song_row = df[df['track_id'] == track_id]
            
            if not song_row.empty:
                artists = song_row['artist(s)_name'].iloc[0].split(', ')
                
                for artist in artists:
                    if artist not in artist_songs:
                        artist_songs[artist] = []
                    
                    if track_id in track_features:
                        artist_songs[artist].append(track_features[track_id])
    
    # Calculate average features for each artist
    for artist, songs in artist_songs.items():
        if songs:
            artist_avg_features[artist] = np.mean(songs, axis=0)
        else:
            artist_avg_features[artist] = np.zeros(len(numerical_features), dtype=np.float32)
    
    # Now create node features for all nodes in the graph
    for node in G.nodes():
        node_type, node_id = node.split('_', 1)  # 'song_X' or 'artist_name'
        
        if node_type == 'song':
            # Extract the track_id from the node name
            try:
                track_id = int(node_id)
                
                # Use the features we stored earlier
                if track_id in track_features:
                    node_features[node] = track_features[track_id]
                else:
                    # If track not found for some reason, use zeros
                    node_features[node] = np.zeros(len(numerical_features), dtype=np.float32)
            except ValueError:
                # Handle case where track_id is not an integer
                node_features[node] = np.zeros(len(numerical_features), dtype=np.float32)
        
        elif node_type == 'artist':
            if node_id in artist_avg_features:
                node_features[node] = artist_avg_features[node_id]
            else:
                node_features[node] = np.zeros(len(numerical_features), dtype=np.float32)
    
    # Create edge features (e.g., weight attributes)
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)  # Default weight is 1.0
        edge_features[(u, v)] = np.array([weight], dtype=np.float32)
    
    print(f"Created features for {len(node_features)} nodes and {len(edge_features)} edges")
    return node_features, edge_features

# Main execution function
def main(file_path='spotify-2023.csv'):
    # Step 1: Preprocess the dataset
    df = preprocess_spotify_data(file_path)
    
    # Step 2: Create graph structure
    G = create_graph_structure(df)
    
    # Step 3: Prepare features for GNN
    node_features, edge_features = prepare_gnn_features(df, G)
    
    # Step 4: Visualize graph (optional)
    visualize_graph(G)
    
    # Step 5: Export data for GNN libraries
    export_for_gnn_libraries(node_features, edge_features, G)
    
    print("\nPreprocessing complete! Data is ready for GNN modeling.")
    return df, G, node_features, edge_features

# If running as a script
if __name__ == "__main__":
    # Replace with actual path to the dataset
    file_path = "spotify-2023.csv"  # or provide the path where the dataset is saved
    main(file_path)