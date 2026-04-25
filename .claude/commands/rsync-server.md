Pull the latest code from the remote server into the local workspace.

Use $ARGUMENTS as the SSH host if provided, otherwise default to "param".

Run exactly this bash command:

rsync -avz --delete --progress --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='checkpoints' --exclude='*.output' ${ARGUMENTS:-param}:~/workspace/vision-coder-openenv/ /Users/amaljoe/Desktop/Workspace/AI/vision-coder-openenv/

After it completes, print a one-line summary: how many files were transferred and total bytes received. If rsync fails, show the error.
