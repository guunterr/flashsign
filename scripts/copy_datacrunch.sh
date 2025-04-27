# args: $1 = IP address, $2 = kernel number, $3 = datetime

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

scp -r src/ root@$1:~/flashsign/ &
scp do_profile.sh root@$1:~/flashsign/ &

wait

echo "Copy complete"