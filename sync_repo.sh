#!/usr/bin/env bash

git clone --bare https://bitbucket.ataccama.com/scm/ai_research/similarity_detection.git
cd similarity_detection.git
git push --mirror https://github.com/chudyandrej/similarity_detection.git
cd ..
rm -rf similarity_detection.git