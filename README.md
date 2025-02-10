# Relational RAG: A RAG over PostgreSQL

This repo implements a Relational RAG proof-of-concept using PostgreSQL. This project can be deployed purely locally, without any usage of any paid api.

- Chat Model: Llama-3.1-8B
- Embedding Model: [Visualized BGE](https://huggingface.co/BAAI/bge-visualized) (Multi-modal)
## Start
### Data Preparation
Go to `data_preparation/README.md` and follow the instructions.

### Setting up the environments
Set the postgresql table and user information in `.env`. 
We recommand to deploy the project in a conda environment with python==3.10. Install the necessary packages by:
```bash
python -m pip install -r requirements.txt
```
### Running the code
1. Build backend and frontend
```bash
# backend
python -m pip install -e src/backend
# frontend
cd src/frontend
npm install
npm run build
```

2. Run the backend and frontend with hot reloading
```bash
# backend
python -m uvicorn fastapi_app:create_app --factory --reload
# frontend, in another window
cd src/frontend
npm run dev
```
Open the browser at `http://localhost:5173/` then will see it.

## License and Acknowledgement

This project is licensed under the MIT License. The frontend parts of the code in this repository are derived from [Azure Samples](https://github.com/Azure-Samples/rag-postgres-openai-python), which also under MIT License. 