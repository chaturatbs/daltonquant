# linux packages and settings
sudo apt install libpng16-dev
sudo apt install python-tk
sudo apt install sqlite3
sudo apt install git
sudo apt install python-pip
sudo pip install --upgrade pip
export DISPLAY=:0.0

# python dependencies for daltonquant
# install specimen-tools and clone to have access to db building script
sudo pip install specimen-tools
git clone https://github.com/josepablocam/specimen-tools.git
sudo pip install -r requirements.txt
# install package locally
sudo pip install .

# PNGQuant dependency
git clone --recursive https://github.com/kornelski/pngquant.git --branch 2.10.2
cd pngquant/
sudo make install
