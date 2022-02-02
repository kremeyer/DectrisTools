python3 -m venv install_test
source install_test/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install DectrisTools
python3 -m DectrisTools.liveview fe80::4ed9:8fff:feca:a8f9 80
rm -R install_test