#!/usr/bin/env bash
# Download the MMPTrack sequences for docs/PROTOCOL_DETECTORS.md into data/mmp2/<tag>/ (git-ignored). One curl at a time.
set -u
B=https://huggingface.co/datasets/hungnguyen190204/MMPTracking/resolve/main/MMPTracking
cd "$(dirname "$0")/.." && mkdir -p data/mmp2
fetch() { # tag split session scene name
  local tag=$1 split=$2 sess=$3 scene=$4 name=$5 d=data/mmp2/$1
  mkdir -p "$d"
  [ -s "$d/calibrations.json" ] || curl -sL -o "$d/calibrations.json" "$B/$split/calibrations/$scene/calibrations.json"
  [ -s "$d/labels.zip" ] || curl -sL -o "$d/labels.zip" "$B/$split/labels/$sess/$name.zip"
  [ -s "$d/topdown.zip" ] || curl -sL -o "$d/topdown.zip" "$B/$split/topdown_labels/$sess/$name.zip"
  curl -sL -C - --retry 10 --retry-delay 3 -o "$d/images.zip" "$B/$split/images/$sess/$name.zip"
  echo "$tag done $(stat -c %s "$d/images.zip")"
}
fetch ft_lobby train 63am lobby lobby_1
fetch ft_office train 63am office office_0
fetch ft_retail train 63am retail retail_0
fetch dev_lobby train 64am lobby lobby_0
fetch dev_office train 64am office office_0
fetch dev_retail train 64am retail retail_1
fetch test_lobby validation 64pm lobby lobby_1
fetch test_office validation 64pm office office_1
fetch test_retail validation 64pm retail retail_1
echo all-done
