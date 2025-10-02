### Steps to install UV on mac

- `brew install uv`  Installs uv
- `uv init myproject`  Creates a new project folder named 'myproject'
- `cd myproject`  Change directory to the project folder
- `uv add -r requirements.txt`  Install required libraries
- `python -m jupyterlab --notebook-dir=<your path to folder> --no-browser --allow-root` Start jupyter lab