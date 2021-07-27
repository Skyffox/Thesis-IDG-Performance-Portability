advisor --collect=survey --project-dir=./data --stackwalk-mode=online --static-instruction-mix -- ./run-gridder.x 1
advisor --collect=tripcounts --project-dir=./data --flop --target-device=gen12_dg1 -- ./run-gridder.x 1
advisor --collect=projection --project-dir=./data --config=gen12_dg1 --no-assume-dependencies 

# only to be run on gpus
#advisor --collect=survey --project-dir=./data2 --profile-gpu -- ./run-gridder.x 1
#advisor --collect=tripcounts --project-dir=./data2 --flop --profile-gpu -- ./run-gridder.x 1
#advisor --report=roofline --gpu --project-dir=./data2 --report-output=roofline.html
