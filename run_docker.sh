#!/bin/bash
set -x

docker run -it   --name tarsier-vision   --privileged   --device /dev/apex_0:/dev/apex_0   --group-add $(getent group apex | cut -d: -f3)   --network host   -v /home/casa/tarsier-vision:/workspace -e CAM_IP=${CAM_IP} -e CAM_PWD=${CAM_PWD} -e TG_TOKEN=${TG_TOKEN} -e CHAT_ID=${CHAT_FRA_ID}  tarsier-vision-image

