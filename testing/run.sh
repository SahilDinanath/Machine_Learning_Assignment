#!/usr/bin/env bash

echo "<<training model>>"
python3 nn.py
echo "<<testing model>>"
python3 classifyall.py
