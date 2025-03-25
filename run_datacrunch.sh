#First argument is IP address


ssh-keygen -R $1
ssh root@$1 '\
    export PATH="/usr/local/cuda/bin:$PATH"; \
    cd flashsign; \
    eval $(ssh-agent -s); \
    ssh-add ~/datacrunch_flashsign_key \ 
    git reset --hard origin/master; \
    git pull origin kernel4; \
    chmod +x do_profile.sh; \
    ./do_profile.sh;'

scp root@$1:~/flashsign/profile.ncu-rep out/profile.ncu-rep
scp root@$1:~/flashsign/output.txt out/output.txt