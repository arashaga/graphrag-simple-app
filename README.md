# GraphRAG Simple Application

This is a simple application containing a FastAPI backend and a Streamlit frontend to showcase the GraphRAG capabilities. The indexing and ingestion of building the Knowledge Graph are not included in this app at this time. You can follow the indexing instructions [here](https://microsoft.github.io/graphrag/posts/get_started/) to build your knowledge graph. We are planning to add this capability in the near future.

## Running the App

1. Clone the GitHub repository.
2. Replace the output folder with your own output folder that is created after your indexing process. Currently, the output folder contains a knowledge graph based on the sample input data, "Alice in Wonderland" book.
3. Install the required packages by running `pip install -r requirements-local.txt`. (You can create a virtual environment for this, so make sure to activate it.)
4. Navigate to the `backend` folder and adjust the output folder path on line 41 in `app.py` to point to your own `output/.../artifacts` folder.
5. Create a `.env` file based on `.env_sample` and replace the variables with your own.
6. Run `python app.py` in the command line to start the backend (ensure that your virtual environment is activated if you've created one).
7. Navigate to the `frontend` folder and run `streamlit run main.py` to start the frontend. A browser window will usually pop up, but if not, go to `http://localhost`.

