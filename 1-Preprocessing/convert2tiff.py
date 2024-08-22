# this script is to convert the original mds images to tiff format
import os
import argparse
import numpy as np
import pandas as pd
from osgeo import gdal
from pathlib import Path
from pma_python import *
from multiprocessing import Pool
import math
import tqdm

_pmaCoreUrl = "http://localhost:54001/"

def convert(dst, slide_path, patient_id):

    save_path = os.path.join(dst, f"{patient_id}.tif")

    # set the target TIFF quality 0-100
    target_quality = 100
    # set the target scale factor to download. One of [1, 2, 4, 8, 16, 32, 64, 128]
    downscale_factor = 1
    # Get the slide information and information about each zoomlevel available

    print("Fetching image info for {0}".format(slide_path))
    slideInfo = core.get_slide_info(slide_path)
    print(slideInfo)
    zoomLevelsInfo = core.get_zoomlevels_dict(slide_path)
    maxLevel = max(zoomLevelsInfo)
    tileSize = slideInfo["TileSize"]
    print("Horizontal Tiles | Vertical Tiles | Total Tiles")
    for level in zoomLevelsInfo:
        tilesX, tilesY, totalTiles = zoomLevelsInfo[level]
        print("{:>16} |{:>15} |{:>12}".format(tilesX, tilesY, totalTiles))

    xresolution = 10000 / slideInfo["MicrometresPerPixelX"]
    yresolution = 10000 / slideInfo["MicrometresPerPixelY"]

    # Create new TIFF file using the GDAL TIFF driver
    # The width and height of the final tiff is based on number of tiles horizontally and vertically.

    # Validate the parameters
    # if target_quality is None or target_quality < 0 or target_quality > 90:
    #    target_quality = 80
    if downscale_factor not in [1, 2, 4, 8, 16, 32, 64, 128]:
        downscale_factor = 1


    maxLevel = max(zoomLevelsInfo)
    powerof2 = int(math.log2(downscale_factor))

    level = maxLevel - powerof2
    level = min(max(level, 0), maxLevel)
    tilesX, tilesY, totalTiles = zoomLevelsInfo[level]

    # We set the region of the image we want to read to set the final tif size accordingly
    tileRegionX = (0, tilesX)
    tileRegionY = (0, tilesY)

    tileSize = 512
    tiff_drv = gdal.GetDriverByName("GTiff")
    # Set the final size
    ds = tiff_drv.Create(
        save_path,
        int((tileRegionX[1] - tileRegionX[0]) * 512),
        int((tileRegionY[1] - tileRegionY[0]) * 512),
        3,
        options=['BIGTIFF=YES',
            'COMPRESS=JPEG', 'TILED=YES', 'BLOCKXSIZE=' + str(tileSize), 'BLOCKYSIZE=' + str(tileSize),
            'JPEG_QUALITY=90', 'PHOTOMETRIC=RGB'
        ])
    descr = "ImageJ=\nhyperstack=true\nimages=1\nchannels=1\nslices=1\nframes=1"
    ds.SetMetadata({ 'TIFFTAG_RESOLUTIONUNIT': '3', 'TIFFTAG_XRESOLUTION': str(int(xresolution / downscale_factor)), 'TIFFTAG_YRESOLUTION': str(int(yresolution / downscale_factor)), 'TIFFTAG_IMAGEDESCRIPTION': descr })


    print("Maximum level = ", maxLevel, ", level = ", level, ", power of 2 = ", powerof2)

    # We read each tile of the final zoomlevel (1:1 resolution) from the server and write it to the resulting TIFF file
    # Then we create the pyramid of the file using BuildOverviews function of GDAL
    tilesX, tilesY, totalTiles = zoomLevelsInfo[level]
    print("Requesting level {}".format(level))

    for x in range(tileRegionX[0], tileRegionX[1]):
        for y in range(tileRegionY[0],tileRegionY[1], 1):  # range of y-axis in which we are interested for this slide

            tile = core.get_tile(slide_path, x, y , level, quality=target_quality)
            arr = np.array(tile, np.uint8)

            # calculate startx starty pixel coordinates based on tile indexes (x,y)
            # for the final tif we want the first tile, i.e. (tileRegionX[0], tileRegionY[0]) ,to be at (0,0) so we need to transform the coordinates
            sx = (x - tileRegionX[0]) * tileSize
            sy = (y - tileRegionY[0]) * tileSize

            ds.GetRasterBand(1).WriteArray(arr[..., 0], sx, sy)
            ds.GetRasterBand(2).WriteArray(arr[..., 1], sx, sy)
            ds.GetRasterBand(3).WriteArray(arr[..., 2], sx, sy)

    print("Please wait while building the pyramid")
    ds.BuildOverviews('average', [pow(2, l) for l in range(1, level)])
    ds = None
    print(f"Slide {patient_id} saved to {save_path}")


def main(args):
    csv = pd.read_csv(args.csv)
    slide_path = []
    for i in range(len(csv)):
        stain = csv.iloc[i]['stain']
        dx = csv.iloc[i]['diagnosis']
        pid = csv.iloc[i]['patient_id']
        cohort = str(csv.iloc[i]['cohort'])
        temp = os.path.join(args.src, cohort, dx, stain, pid, '1.mds')
        assert os.path.exists(temp), f'No such file: {temp}'
        slide_path.append(temp)
    with Pool(args.workers) as p:
        p.starmap(convert, [(args.dst, slide_path[i], csv.iloc[i]['patient_id']) for i in range(len(csv))])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MDS to TIFF')
    parser.add_argument('--src', type=str, default='/mnt/zhen_chen/AIEC_rawdata')
    parser.add_argument('--dst', type=str, default='/mnt/zhen_chen/AIEC_tongji')
    parser.add_argument('--csv', type=str, default='./tongji_samples.csv')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    main(args)