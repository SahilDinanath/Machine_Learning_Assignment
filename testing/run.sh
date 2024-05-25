#!/usr/bin/env bash

echo "<<Preprocessing Data>>"
time python3 preprocessor.py
echo "<<training model>>"
time python3 nn.py
echo ""
# echo "<<testing model>>"
# time python3 classifyall.py
