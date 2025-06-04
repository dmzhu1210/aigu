# AIGU
A Framework for Vulnerability Detection

### Prerequisites
- Download the required datasets: Devign, Reveal, and DiverseVul.
- Train the word embedding model using `word2vec.py`.

### Installation Steps
1. **Data Preparation**
   - Run `init_data.py` in the `process` directory to initialize the dataset.
   - Execute `check_node.py` to validate and process the dataset.

2. **Input Construction**
   - Run `smcpgg.py` in the `process` directory to construct the required inputs.
   - Modify the path to the Word2Vec model in `tools.py`.

3. **Configuration**
   - Update the dataset paths in `dataset.py`.

4. **Training and Evaluation**
   - Run `main.py` to train the model and obtain results.
