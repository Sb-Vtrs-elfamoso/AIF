# Movie poster Recommender System

## Requirements :
- Docker engine is installed on your computer. If it not the case, please follow these [instructions](https://docs.docker.com/engine/install/)
- have a stable internet connection (for packages installation)

## User guide :
1. download or clone this respository
2. go inside the folder **recomender_system** with a terminal
3. run : `sudo docker-compose up -d`
4. in your brower open the [web-app](http://172.18.0.1:7860/)
5. you can now upload any image and submit it, the system will recommend you 5 similar movie posters
6. to close all the containers and remove their images run : `sudo docker-compose down --rmi all`
if you want to keep the images run : `sudo docker-compose down`
