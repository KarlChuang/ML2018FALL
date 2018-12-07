if [ "$(python3 --version | head -n1 | cut -d " " -f2 | cut -d "." -f2)" == 6 ]; then
    python3 train.py $1 $2
else
    python3.6 train.py $1 $2
fi
