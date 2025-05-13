
# GNN-based Music Recommender System

A music recommendation system built using Graph Neural Networks (GNN) to provide personalized song suggestions based on user preferences and music features.

## 📝 Overview

This project implements a recommendation system for music using Graph Neural Networks (GNNs). By representing songs, artists, and users as nodes in a graph with their relationships as edges, the system can capture complex patterns and similarities that might not be apparent in traditional recommendation approaches.

## ✨ Features

- **Graph-based Music Recommendations**: Utilizes the power of Graph Neural Networks to recommend music based on complex relationships.
- **Song Similarity**: Finds similar songs based on audio features and user listening patterns.
- **Web Interface**: User-friendly interface to interact with the recommendation system.
- **Integration with Music Data**: Works with Spotify's 2023 dataset.
- **Audio Preview**: Listen to song previews when available.

## 🛠️ Technologies Used

- **Python (60.9%)**: Core GNN implementation and data processing.
- **HTML (39.1%)**: Web interface.
- **[PyTorch](https://pytorch.org/)**: For building and training the GNN model.
- **[Django](https://www.djangoproject.com/)**: Web framework for the application.
- **Spotify Data**: Dataset for music information and features.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shialley/GNN-based-Music-Recommender-System.git
   cd GNN-based-Music-Recommender-System
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r web/requirements.txt
   ```

4. Prepare the database:
   ```bash
   cd web
   python manage.py migrate
   ```

### Running the Application

1. Start the Django server:
   ```bash
   python manage.py runserver
   ```

2. Open your browser and navigate to `http://127.0.0.1:8000/`

## 📊 Project Structure

```
GNN-based-Music-Recommender-System/
├── GNN_RecommenderSystem.py      # Core GNN model implementation
├── dataprepocessing.py           # Data preparation scripts
├── gnn_data/                     # Processed graph data
├── processed_data/               # Processed dataset files
├── spotify-2023.csv              # Original dataset
├── music_gnn_model.pt            # Trained model
├── web/                          # Web application
│   ├── manage.py                 # Django management script
│   ├── recommender/              # Main app
│   │   ├── models.py             # Data models
│   │   ├── views.py              # Views and endpoints
│   │   ├── services.py           # Recommendation service
│   │   └── templates/            # HTML templates
│   ├── static/                   # Static files (CSS, JS, audio)
│   └── requirements.txt          # Python dependencies
└── README.md                     # This file
```

## 📸 Screenshots

<details>
<summary>View Screenshots</summary>

### 网页界面展示
| 主页 | 搜索结果 | 推荐详情 |
|------|----------|----------|
| ![主页界面](GNN-based-Music-Recommender-System/blob/main/Web_illustration/home_page.png) | ![搜索结果](GNN-based-Music-Recommender-System/blob/main/Web_illustration/search_results.png) | ![推荐详情](GNN-based-Music-Recommender-System/blob/main/Web_illustration/song_recommendation_details.png) |

</details>

## 🧠 How It Works

1. **Data Processing**: Music data is processed and transformed into a graph structure with songs, artists, and features as nodes.
2. **GNN Model**: A Graph Neural Network is trained to learn embeddings that capture the relationships between different entities.
3. **Recommendation Generation**: When a user selects a song or artist, the system uses the trained GNN to find the most similar items in the embedding space.
4. **Web Interface**: The Django web application provides a user-friendly interface to interact with the recommendation system.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Acknowledgments

- Spotify dataset for providing comprehensive music information.
- PyTorch and PyTorch Geometric teams for their amazing frameworks.
- All contributors who have helped with the development of this project.

---

*Made with ❤️ by Shialley*
