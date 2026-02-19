# Flask API Project

A simple Flask API application with health check endpoints.

## Running with Docker

Build the Docker image:

```bash
docker build -t flask-api .
```

Run the container:

```bash
docker run -p 5000:5000 flask-api
```

The API will be available at `http://localhost:5000`

## Running Locally (without Docker)

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

## Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint
