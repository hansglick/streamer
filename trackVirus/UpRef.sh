cd /home/osboxes/proj/streamer/trackVirus

/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackVirus/BuildBatch.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackVirus/BuildBatch.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackVirus/BuildRef.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackVirus/BuildRef.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackVirus/BatchBest.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackVirus/BatchBest.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackVirus/Graph.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackVirus/Graph.py


/home/osboxes/anaconda3/bin/jupyter nbconvert --to script /home/osboxes/proj/streamer/trackVirus/ComputeTFIDF.ipynb
/home/osboxes/anaconda3/envs/twitter/bin/python /home/osboxes/proj/streamer/trackVirus/ComputeTFIDF.py
