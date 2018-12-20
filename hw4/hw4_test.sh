mkdir model
wget 'https://www.dropbox.com/s/hsue9yllgy7o3oe/model_9.model?dl=1'
mv model_9.model?dl=1 ./model/model_9.model
wget 'https://www.dropbox.com/s/amypm597xzq08r9/embedding250.model?dl=1'
mv embedding250.model?dl=1 ./model/embedding250.model

python3.6 test_final.py ./model/model_9.model $1 $2 $3
