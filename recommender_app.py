import os
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import our recommender model
from GNN_RecommenderSystem import MusicGNN, load_processed_data, generate_recommendations

class MusicRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GNN Music Recommender")
        self.root.geometry("800x600")
        
        # Load model and data
        self.load_model_and_data()
        
        # Create UI elements
        self.create_ui()
    
    def load_model_and_data(self):
        try:
            # Load processed data
            self.data, self.node_types = load_processed_data("processed_data")
            
            # Get song nodes
            self.song_nodes = [(idx, name) for idx, (node_type, name) in self.node_types.items() 
                             if node_type == 'song']
            self.song_names = [name for _, name in self.song_nodes]
            self.song_indices = [idx for idx, _ in self.song_nodes]
            
            # Load trained model
            model_path = "music_gnn_model.pt"
            if os.path.exists(model_path):
                self.model = MusicGNN(num_node_features=self.data.x.size(1))
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                print("Model loaded successfully!")
            else:
                messagebox.showerror("Error", "Model file not found. Please train the model first.")
                self.root.destroy()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model and data: {str(e)}")
            self.root.destroy()
    
    def create_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create song selection section
        ttk.Label(main_frame, text="Select a Song:", font=("Arial", 12, "bold")).pack(pady=(10, 5))
        
        # Create search box
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_songs)
        
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        
        # Create a frame to contain both listbox and scrollbar
        listbox_frame = ttk.Frame(main_frame)
        listbox_frame.pack(pady=5, fill=tk.X)

        # Create song listbox
        self.song_listbox = tk.Listbox(listbox_frame, height=10, width=70)
        self.song_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add scrollbar to listbox
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.song_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.song_listbox.config(yscrollcommand=scrollbar.set)

        # Assuming this is the incomplete line
        self.song_list = []  # Initialize the song list

        # Populate with sample data or actual data
        self.populate_song_list()

        # Add a button to get recommendations for the selected song
        self.recommend_button = ttk.Button(main_frame, text="Get Recommendations", 
                                           command=self.get_recommendations)
        self.recommend_button.pack(pady=10)

        # Add result display area
        self.result_frame = ttk.LabelFrame(main_frame, text="Recommendations")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        # Add a listbox for recommendations
        self.recommendations_listbox = tk.Listbox(self.result_frame, height=10, width=70)
        self.recommendations_listbox.pack(pady=5, fill=tk.BOTH, expand=True)

    def filter_songs(self, *args):
        search_term = self.search_var.get().lower()
        self.song_listbox.delete(0, tk.END)
        
        for song in self.song_list:
            if search_term in song.lower():
                self.song_listbox.insert(tk.END, song)