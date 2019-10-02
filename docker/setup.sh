#!/bin/bash

cd /opt/lab

for file in /opt/lab/setup_steps/*.sh
do
	$file
done
