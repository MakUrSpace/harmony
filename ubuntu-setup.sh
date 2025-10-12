curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -

curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list 

sudo apt update && sudo apt upgrade -y && sudo apt install -y python3 python3-pip python3-dev graphviz libgraphviz-dev pkg-config nodejs ngrok

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt