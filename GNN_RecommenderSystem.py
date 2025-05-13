import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

class MusicGNN(torch.nn.Module):
    """
    Music Recommendation GNN model using a Graph Convolutional Network (GCN) architecture
    """
    def __init__(self, num_node_features, hidden_channels=64, out_channels=32):
        super(MusicGNN, self).__init__()
        # First Graph Convolutional layer
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        # Second Graph Convolutional layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Third Graph Convolutional layer for generating embeddings
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
        # Layers for link prediction
        self.link_predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
    
    def encode(self, x, edge_index):
        # Apply GNN layers to generate node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        # For each edge, concatenate the embeddings of both nodes
        row, col = edge_index
        z_src = z[row]
        z_dst = z[col]
        edge_features = torch.cat([z_src, z_dst], dim=1)
        
        # Predict link probability
        return self.link_predictor(edge_features)
    
    def forward(self, x, edge_index, edge_label_index):
        # Get node embeddings
        z = self.encode(x, edge_index)
        
        # Predict link probabilities for the edges in edge_label_index
        link_pred = self.decode(z, edge_label_index)
        return link_pred, z

def load_processed_data(data_dir="processed_data"):
    """
    Load the processed data that was exported by the preprocessing script
    """
    print("Loading processed data...")
    
    # Load node features
    node_features = np.load(f"{data_dir}/node_features.npy")
    
    # Load node mapping
    node_mapping_df = pd.read_csv(f"{data_dir}/node_mapping.txt", sep="\t")
    
    # Load edges
    edges_df = pd.read_csv(f"{data_dir}/edges.txt", sep="\t")
    
    # Create PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([edges_df["source_idx"].values, 
                               edges_df["target_idx"].values], dtype=torch.long)
    edge_attr = torch.tensor(edges_df["weight"].values, dtype=torch.float).reshape(-1, 1)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Create mappings for song and artist nodes
    node_types = {}
    for _, row in node_mapping_df.iterrows():
        node_idx = row["index"]
        node_type = row["node_type"]
        node_name = row["node_name"]
        node_types[node_idx] = (node_type, node_name)
    
    print(f"Loaded graph with {data.num_nodes} nodes and {data.num_edges} edges")
    return data, node_types

def sample_negative_edges(edge_index, num_nodes, num_samples, existing_edges):
    """Generate negative samples (edges that don't exist)"""
    neg_edges = []
    while len(neg_edges) < num_samples:
        # Sample random node pairs
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        # Skip self-loops and existing edges
        if src == dst or (src, dst) in existing_edges or (dst, src) in existing_edges:
            continue
            
        neg_edges.append([src, dst])
        existing_edges.add((src, dst))
        existing_edges.add((dst, src))  # Add both directions if undirected
        
    return torch.tensor(neg_edges, dtype=torch.long).t()

def create_train_test_split(data, test_ratio=0.2, val_ratio=0.1):
    """
    Split the edges into training, validation, and test sets
    """
    print("Creating train/validation/test splits...")
    
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    
    # Convert edge_index to edge list format for easier processing
    edge_list = edge_index.t().tolist()
    
    # Create a set of existing edges
    edge_set = set(map(tuple, edge_list))
    
    # Split edges into train, val, and test
    num_edges = len(edge_list)
    num_test = int(num_edges * test_ratio)
    num_val = int(num_edges * val_ratio)
    num_train = num_edges - num_test - num_val
    
    # Convert edge list to a numpy array and shuffle
    edge_array = np.array(edge_list)
    np.random.shuffle(edge_array)
    
    # Split the edges
    train_edges = edge_array[:num_train]
    val_edges = edge_array[num_train:num_train + num_val]
    test_edges = edge_array[num_train + num_val:]
    
    # Convert back to PyTorch tensors
    train_edge_index = torch.tensor(train_edges.T, dtype=torch.long)
    val_edge_index = torch.tensor(val_edges.T, dtype=torch.long)
    test_edge_index = torch.tensor(test_edges.T, dtype=torch.long)
    
    # Create negative samples
    val_neg_edge_index = sample_negative_edges(val_edge_index, num_nodes, 
                                              val_edges.shape[0], edge_set)
    test_neg_edge_index = sample_negative_edges(test_edge_index, num_nodes, 
                                               test_edges.shape[0], edge_set.union(set(map(tuple, val_edges))))
    
    # Create labels (1 for positive edges, 0 for negative edges)
    val_labels = torch.cat([torch.ones(val_edges.shape[0]), 
                            torch.zeros(val_edges.shape[0])], dim=0)
    test_labels = torch.cat([torch.ones(test_edges.shape[0]), 
                             torch.zeros(test_edges.shape[0])], dim=0)
    
    print(f"Split edges into {num_train} train, {num_val} validation, and {num_test} test edges")
    
    return train_edge_index, (val_edge_index, val_neg_edge_index, val_labels), (test_edge_index, test_neg_edge_index, test_labels)

def train_model(model, data, train_edge_index, val_data, num_epochs=100, learning_rate=0.01):
    """
    Train the GNN model for link prediction
    """
    print("Training model...")
    
    # Extract validation data
    val_edge_index, val_neg_edge_index, val_labels = val_data
    val_edge_label_index = torch.cat([val_edge_index, val_neg_edge_index], dim=1)
    
    # Generate negative samples for training
    print("Generating negative training samples...")
    edge_set = set(map(tuple, train_edge_index.t().tolist()))
    train_neg_edge_index = sample_negative_edges(
        train_edge_index, data.num_nodes, train_edge_index.size(1), edge_set
    )
    
    # Combine positive and negative edges for training
    train_edge_label_index = torch.cat([train_edge_index, train_neg_edge_index], dim=1)
    train_labels = torch.cat([
        torch.ones(train_edge_index.size(1)), 
        torch.zeros(train_neg_edge_index.size(1))
    ]).unsqueeze(1)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    best_val_auc = 0
    best_model_state = None
    
    loss_history = []
    val_auc_history = []
    
    for epoch in tqdm(range(num_epochs)):
        # Forward pass with both positive and negative samples
        optimizer.zero_grad()
        link_logits, _ = model(data.x, data.edge_index, train_edge_label_index)
        
        # Binary classification loss using both positive and negative examples
        loss = F.binary_cross_entropy_with_logits(link_logits, train_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Validation
        if epoch % 5 == 0:
            val_auc = evaluate_model(model, data, val_edge_label_index, val_labels)
            val_auc_history.append(val_auc)
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {key: value.cpu() for key, value in model.state_dict().items()}
            
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Val AUC = {val_auc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, num_epochs, 5), val_auc_history)
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    
    return model, best_val_auc

def calculate_precision_at_k(recommendations, relevant_items, k=10):
    """
    Calculate Precision@K
    
    Args:
        recommendations: list of recommended item ids
        relevant_items: set of relevant item ids
        k: number of recommendations to consider
    
    Returns:
        precision@k value
    """
    # Consider only top-k recommendations
    top_k_recommendations = recommendations[:k]
    
    # Count relevant items in top-k
    num_relevant = sum(1 for item in top_k_recommendations if item in relevant_items)
    
    # Calculate precision
    return num_relevant / min(k, len(recommendations)) if min(k, len(recommendations)) > 0 else 0.0

def calculate_recall_at_k(recommendations, relevant_items, k=10):
    """
    Calculate Recall@K
    
    Args:
        recommendations: list of recommended item ids
        relevant_items: set of relevant item ids
        k: number of recommendations to consider
    
    Returns:
        recall@k value
    """
    # Consider only top-k recommendations
    top_k_recommendations = recommendations[:k]
    
    # Count relevant items in top-k
    num_relevant_recommended = sum(1 for item in top_k_recommendations if item in relevant_items)
    
    # Calculate recall
    return num_relevant_recommended / len(relevant_items) if len(relevant_items) > 0 else 0.0

def calculate_mrr(recommendations, relevant_items):
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        recommendations: list of recommended item ids
        relevant_items: set of relevant item ids
    
    Returns:
        MRR value
    """
    # Find the rank of the first relevant item
    for i, item in enumerate(recommendations):
        if item in relevant_items:
            return 1.0 / (i + 1)
    
    # No relevant items found
    return 0.0

def evaluate_model(model, data, edge_label_index, labels, k_values=[5, 10, 20], node_types=None):
    """
    Evaluate the model using multiple metrics
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        # Calculate AUC
        link_logits, _ = model(data.x, data.edge_index, edge_label_index)
        link_probs = torch.sigmoid(link_logits)
        auc = roc_auc_score(labels.numpy(), link_probs.numpy())
        results['auc'] = auc
        
        # For other metrics, we need node embeddings
        embeddings = model.encode(data.x, data.edge_index)
        
        # Calculate Precision@K, Recall@K, MRR
        # This requires additional processing to identify relevant items for each node
        if node_types:
            # Create dictionary of song indices
            song_indices = [idx for idx, (node_type, _) in node_types.items() if node_type == 'song']
            
            # For each song, calculate metrics
            precision_at_k = {k: [] for k in k_values}
            recall_at_k = {k: [] for k in k_values}
            mrr_list = []
            
            for song_idx in song_indices[:100]:  # Limit to 100 songs for efficiency
                # Get relevant items from ground truth
                # For this example, we'll use nodes connected in the original graph
                relevant_items = set()
                edge_index = data.edge_index
                
                # Find nodes connected to this song in the original graph
                for i in range(edge_index.shape[1]):
                    if edge_index[0, i].item() == song_idx:
                        target = edge_index[1, i].item()
                        if target in song_indices and target != song_idx:
                            relevant_items.add(target)
                    elif edge_index[1, i].item() == song_idx:
                        source = edge_index[0, i].item()
                        if source in song_indices and source != song_idx:
                            relevant_items.add(source)
                
                if len(relevant_items) == 0:
                    continue  # Skip songs without relevant items
                
                # Get recommendations for this song
                song_embedding = embeddings[song_idx].unsqueeze(0)
                similarities = F.cosine_similarity(song_embedding, embeddings)
                
                # Get ranked recommendations (excluding self)
                song_similarities = [(idx, similarities[idx].item()) for idx in song_indices if idx != song_idx]
                ranked_songs = [idx for idx, _ in sorted(song_similarities, key=lambda x: x[1], reverse=True)]
                
                # Calculate metrics
                for k in k_values:
                    precision_at_k[k].append(calculate_precision_at_k(ranked_songs, relevant_items, k))
                    recall_at_k[k].append(calculate_recall_at_k(ranked_songs, relevant_items, k))
                
                mrr_list.append(calculate_mrr(ranked_songs, relevant_items))
            
            # Average the metrics
            for k in k_values:
                if precision_at_k[k]:
                    results[f'precision@{k}'] = np.mean(precision_at_k[k])
                if recall_at_k[k]:
                    results[f'recall@{k}'] = np.mean(recall_at_k[k])
            
            if mrr_list:
                results['mrr'] = np.mean(mrr_list)
    
    return results

def generate_recommendations(model, data, node_types, user_song_idx, top_n=10):
    """
    Generate song recommendations for a given song
    """
    model.eval()
    
    # Get all node embeddings
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
    
    # Get the embedding of the input song
    song_embedding = embeddings[user_song_idx].unsqueeze(0)
    
    # Calculate similarity to all other nodes
    similarities = F.cosine_similarity(song_embedding, embeddings)
    
    # Find the top similar songs (excluding the input song and non-song nodes)
    song_indices = [idx for idx, (node_type, _) in node_types.items() 
                    if node_type == 'song' and idx != user_song_idx]
    
    song_similarities = [(idx, similarities[idx].item()) for idx in song_indices]
    top_songs = sorted(song_similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Return the recommended songs
    recommendations = []
    for idx, sim in top_songs:
        song_name = node_types[idx][1]
        recommendations.append((song_name, sim))
    
    return recommendations

def main():
    # Step 1: Load the preprocessed data
    data_dir = "processed_data"
    data, node_types = load_processed_data(data_dir)
    
    # Step 2: Create train/validation/test splits
    train_edge_index, val_data, test_data = create_train_test_split(data)
    
    # Step 3: Create and train the model
    model = MusicGNN(num_node_features=data.x.size(1))
    model, best_val_auc = train_model(model, data, train_edge_index, val_data, num_epochs=200)
    
    # Step 4: Evaluate on test data with multiple metrics
    test_edge_index, test_neg_edge_index, test_labels = test_data
    test_edge_label_index = torch.cat([test_edge_index, test_neg_edge_index], dim=1)
    
    test_results = evaluate_model(model, data, test_edge_label_index, test_labels, 
                                 k_values=[5, 10, 20], node_types=node_types)
    
    print("\nModel Evaluation Results:")
    print(f"Test AUC: {test_results['auc']:.4f}")
    
    # Print additional metrics if available
    if 'precision@10' in test_results:
        print(f"Precision@10: {test_results['precision@10']:.4f}")
    if 'recall@10' in test_results:
        print(f"Recall@10: {test_results['recall@10']:.4f}")
    if 'mrr' in test_results:
        print(f"Mean Reciprocal Rank: {test_results['mrr']:.4f}")
    
    # Step 5: Save the model
    torch.save(model.state_dict(), "music_gnn_model.pt")
    
    # Step 6: Example of generating recommendations
    # Find a song node to use as an example
    song_nodes = [(idx, name) for idx, (node_type, name) in node_types.items() if node_type == 'song']
    if song_nodes:
        example_song_idx, example_song_name = song_nodes[0]
        print(f"\nGenerating recommendations for: {example_song_name}")
        
        recommendations = generate_recommendations(model, data, node_types, example_song_idx, top_n=10)
        
        print("\nTop 10 Recommendations:")
        for i, (song_name, similarity) in enumerate(recommendations, 1):
            print(f"{i}. {song_name} (Similarity: {similarity:.4f})")

if __name__ == "__main__":
    main()