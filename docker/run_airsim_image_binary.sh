DOCKER_IMAGE_NAME=$1

# get the base directory name of the unreal binary's shell script
# we'll mount this volume while running docker container
UNREAL_BINARY_PATH=$(dirname $(greadlink -f $2))
ECHO $UNREAL_BINARY_PATH
UNREAL_BINARY_SHELL_ABSPATH=$(greadlink -f $2)
ECHO $UNREAL_BINARY_SHELL_ABSPATH

# this block is for running X apps in docker
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi



# this are the first (maximum) four arguments which the user specifies:
# ex: ./run_airsim_image.sh /PATH/TO/UnrealBinary/UnrealBinary.sh -windowed -ResX=1080 -ResY=720
# we save them in a variable right now:  
UNREAL_BINARY_COMMAND="$UNREAL_BINARY_SHELL_ABSPATH $3 $4 $5"


# now, let's mount the user directory which points to the unreal binary (UNREAL_BINARY_PATH)
# set the environment varible SDL_VIDEODRIVER to SDL_VIDEODRIVER_VALUE
# and tell the docker container to execute UNREAL_BINARY_COMMAND
docker run -it \
    -v $(pwd)/settings.json:/home/airsim_user/Documents/AirSim/settings.json \
    -v $UNREAL_BINARY_PATH:$UNREAL_BINARY_PATH \
    -e SDL_VIDEODRIVER=offscreen\
    -e SDL_HINT_CUDA_DEVICE='0' \
    --net=host \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --rm \
    $DOCKER_IMAGE_NAME \
    /bin/bash -c "$UNREAL_BINARY_COMMAND"