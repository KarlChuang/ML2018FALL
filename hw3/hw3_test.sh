wget 'https://www.dropbox.com/s/df0kjpn3fey62gp/model_8.hs?dl=1'
mv model_8.hs?dl=1 ./model/model_final.hs

if [ "$(python3 --version | head -n1 | cut -d " " -f2 | cut -d "." -f2)" == 6 ]; then
    python3 test.py $1 $2
else
    python3.6 test.py $1 $2
fi
