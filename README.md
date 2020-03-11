# rebuild_tmp

python3 -m rebuild_tmp.data_collection.collect_data -gp -pl pooling -num 10 -shuffle 1

python3 -m rebuild_tmp.data_collection.collect_data -gp -ep -pp -pl pooling -num 10 -shuffle 1 -d 1080ti
