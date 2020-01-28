# rsync -ravz --progress . ylou@cmslpc-sl7.fnal.gov:~/caffe --copy-links
rsync -e "ssh -i ~/louyu27-aws-us-west.pem" -ravz --progress . ylou@cmslpc-sl7.fnal.gov:~/private/ --copy-links
