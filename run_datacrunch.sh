#First argument is IP address, second is kernel number


ssh-keygen -R $1
ssh root@$1 "\
    export PATH="/usr/local/cuda/bin:$PATH"; \
    cd flashsign; \
    eval \$(ssh-agent -s); \
    ssh-add ~/flashsign_key; \ 
    git reset --hard origin/master; \
    git pull origin kernel6; \
    chmod +x do_profile.sh; \
    ./do_profile.sh $2;"

datetime=$(date +%m-%d_%H:%M:%S)
echo $datetime

mkdir profiles/$2
mkdir profiles/$2/$datetime
mkdir profiles/$2/$datetime/kernels
scp root@$1:~/flashsign/out/profile* profiles/$2/$datetime/ &
scp root@$1:~/flashsign/out/test* profiles/$2/$datetime/ &
scp root@$1:~/flashsign/out/benchmark* profiles/$2/$datetime/ &
scp root@$1:~/flashsign/out/compiler* profiles/$2/$datetime/ &
scp root@$1:~/flashsign/kernels/* profiles/$2/$datetime/kernels/ &