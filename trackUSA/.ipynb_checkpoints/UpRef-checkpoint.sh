cd /home/osboxes/proj/streamer/trackUSA

/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackUSA/BuildBatch.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackUSA/BuildBatch.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackUSA/BuildRef.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackUSA/BuildRef.py
