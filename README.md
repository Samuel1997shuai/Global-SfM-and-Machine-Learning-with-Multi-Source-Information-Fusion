1.To build global Sfm, first install COLMAP dependencies and then build peoject using the following commands:

mkdir build

cd build

cmake .. -GNinja

ninja && ninja install

2.After installation, one can run peoject by (starting from a database)

peoject mapper --database_path DATABASE_PATH --output_path OUTPUT_PATH --image_path IMAGE_PATH



Note:

peoject depends on two external libraries - COLMAP and PoseLib. With the default setting, the library is built automatically by peoject via . However, if a self-installed version is preferred, one can also disable the and CMake options.FetchContentFETCH_COLMAPFETCH_POSELIB
To use , the minimum required version of is 3.28. If a self-installed version is used, can be downgraded to 3.10.FetchContentcmakecmake
