cd /home/osboxes/proj/streamer/trackUSA

/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackUSA/BuildBatch.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackUSA/BuildBatch.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackUSA/BuildRef.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackUSA/BuildRef.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackUSA/BatchBest.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackUSA/BatchBest.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackUSA/Graph.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackUSA/Graph.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackUSA/ComputeTFIDF.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackUSA/ComputeTFIDF.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackUSA/Prerequisites.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackUSA/Prerequisites.py
