#First argument is IP address, second is kernel number


ssh-keygen -R $1
ssh root@$1 '\
    export PATH="/usr/local/cuda/bin:$PATH"; \
    cd flashsign; \
    eval $(ssh-agent -s); \
    ssh-add ~/flashsign_key; \ 
    git reset --hard origin/master; \
    git pull origin kernel5; \
    chmod +x do_profile.sh; \
    ./do_profile.sh $2;'

scp root@$1:~/flashsign/out/profile* out/
scp root@$1:~/flashsign/out/test*.txt out/
scp root@$1:~/flashsign/out/benchmark*.txt out/