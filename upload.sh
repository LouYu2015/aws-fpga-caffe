rsync -e "ssh -i ~/louyu27-aws-us-west.pem" -ravz --progress . ubuntu@ec2-54-202-149-245.us-west-2.compute.amazonaws.com:/home/ubuntu/project --copy-links
