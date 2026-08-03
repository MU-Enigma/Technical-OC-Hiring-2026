#!/bin/bash
# Downloads the BSDS500 dataset (502 natural photographs) used to develop
# and test this project, and flattens it into source_images/.
#
# Citation: P. Arbelaez, M. Maire, C. Fowlkes and J. Malik,
# "Contour Detection and Hierarchical Image Segmentation,"
# IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.
#
# If you already have your own folder of images, you don't need this script --
# just place them in source_images/ directly (any .jpg/.png/.bmp, at least
# hr_patch_size pixels in both dimensions).
set -e
cd "$(dirname "$0")/.."

echo "Downloading BSDS500 from GitHub mirror..."
curl -sL -o /tmp/bsds500.zip https://codeload.github.com/BIDS/BSDS500/zip/refs/heads/master

echo "Extracting..."
rm -rf /tmp/bsds500_full
unzip -q /tmp/bsds500.zip -d /tmp/bsds500_full

mkdir -p source_images
find /tmp/bsds500_full -path "*/images/*" -name "*.jpg" -exec cp {} source_images/ \;

count=$(ls source_images | wc -l)
echo "Done. $count images in source_images/"

rm -f /tmp/bsds500.zip
rm -rf /tmp/bsds500_full
