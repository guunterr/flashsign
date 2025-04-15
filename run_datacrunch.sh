#First argument is IP address, second is kernel number
datetime=$(date +%m-%d_%H:%M:%S)
echo $datetime

ssh-keygen -R $1
ssh root@$1 "\
    cd flashsign; \
    eval \$(ssh-agent -s); \
    ssh-add ~/flashsign_key; \ 
    git reset --hard origin/master; \
    git pull origin master; \
    rm -rf out/*; \
    rm -rf src/*; \
    rm -rf do_profile.sh;"

scp src/* root@$1:~/flashsign/src/  &
scp do_profile.sh root@$1:~/flashsign/ &

wait

ssh root@$1 "\
    cd flashsign; \
    export PATH="/usr/local/cuda/bin:\$PATH"; \
    chmod +x do_profile.sh; \
    ./do_profile.sh $2 $datetime;"


mkdir profiles/$2
mkdir profiles/$2/$datetime
mkdir profiles/$2/$datetime/kernels
scp root@$1:~/flashsign/out/profile* profiles/$2/$datetime/ &
scp root@$1:~/flashsign/out/test* profiles/$2/$datetime/ &
scp root@$1:~/flashsign/out/benchmark* profiles/$2/$datetime/ &
scp root@$1:~/flashsign/out/compiler* profiles/$2/$datetime/ &
scp root@$1:~/flashsign/src/kernels/* profiles/$2/$datetime/kernels/ &
wait
echo "Profile complete\n"