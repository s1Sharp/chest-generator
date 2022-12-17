How to set up environment
Linux:
```
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

Windows:
```
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For deactivate the Python venv:
```
deactivate
Update requirements
pip freeze > requirements.txt
```

Usage docker compose
```sh
docker-compose up -d --build
```

Usage docker pull
```sh
docker pull s1sharp/image-generator
```

Usage docker run
```sh
docker run -p 5000:5000 -d s1sharp/chest-generator:latest python web_app/app.py release
```

Usage web application
```sh
python -m webbrowser "http://localhost:5000/"
```
